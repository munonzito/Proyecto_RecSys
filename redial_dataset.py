import json
from typing import List, Dict, Tuple
import re

class ReDiALDataset:
    """Clase para cargar y procesar el dataset ReDiAL"""
    
    def __init__(self, data_path: str, split: str = 'test'):
        """
        Args:
            data_path: ruta al directorio con los archivos (ej: './redial_data' o '.')
            split: 'train', 'validation', o 'test'
        """
        self.data_path = data_path
        self.split = split
        self.conversations = []
        self.load_data()
    
    def load_data(self):
        """Carga las conversaciones del archivo correspondiente"""
        # Mapeo de nombres
        file_mapping = {
            'train': 'train_data.jsonl',
            'validation': 'valid_data.jsonl',
            'test': 'test_data.jsonl'
        }
        
        file_name = file_mapping[self.split]
        
        # Intentar con y sin subdirectorio
        try:
            file_path = f"{self.data_path}/{file_name}"
            with open(file_path, 'r') as f:
                test_line = f.readline()
        except FileNotFoundError:
            file_path = file_name
        
        print(f"Loading {self.split} data from {file_path}...")
        
        with open(file_path, 'r') as f:
            for line in f:
                conv = json.loads(line)
                self.conversations.append(conv)
        
        print(f"Loaded {len(self.conversations)} conversations")
    
    def format_conversation_context(self, conversation: Dict,
                                    n_messages: int) -> str:
        """
        Formatea los mensajes en un string legible para el LLM
        
        Args:
            conversation: conversación completa con metadata
            n_messages: cuántos mensajes incluir
            
        Returns:
            string con la conversación formateada
        """
        messages = conversation['messages']
        movie_mentions = conversation['movieMentions']
        
        # Identificar quién es el seeker (iniciador) y quién el recommender
        seeker_id = conversation['initiatorWorkerId']
        recommender_id = conversation['respondentWorkerId']
        
        formatted = []
        
        for msg in messages[:n_messages]:
            # Mapear correctamente según el workerId
            sender_id = msg['senderWorkerId']
            
            if sender_id == seeker_id:
                sender = "User"  # El que busca recomendaciones
            elif sender_id == recommender_id:
                sender = "Recommender"  # El que recomienda
            else:
                sender = f"Speaker{sender_id}"  # fallback
            
            text = msg['text']
            
            # Reemplazar referencias @ID con nombres de películas
            for movie_id, movie_title in movie_mentions.items():
                text = text.replace(f"@{movie_id}", f'"{movie_title}"')
            
            formatted.append(f"{sender}: {text}")
        
        return "\n".join(formatted)
    
    def extract_ground_truth(self, conversation: Dict, 
                            from_message_idx: int) -> List[str]:
        """
        Extrae las películas que fueron sugeridas después del contexto
        
        Args:
            conversation: diccionario con la conversación completa
            from_message_idx: desde qué mensaje extraer ground truth
            
        Returns:
            lista de títulos de películas sugeridas
        """
        messages = conversation['messages'][from_message_idx:]
        movie_mentions = conversation['movieMentions']
        recommender_id = conversation['respondentWorkerId']
        
        # Extraer IDs de películas mencionadas por el RECOMMENDER después del contexto
        mentioned_ids = set()
        for msg in messages:
            # Solo considerar mensajes del recommender
            if msg['senderWorkerId'] == recommender_id:
                text = msg['text']
                ids = re.findall(r'@(\d+)', text)
                mentioned_ids.update(ids)
        
        # Filtrar solo las que fueron sugeridas (suggested=1)
        ground_truth = []
        
        # Revisar en respondentQuestions
        respondent_questions = conversation.get('respondentQuestions', {})
        for movie_id in mentioned_ids:
            if movie_id in respondent_questions:
                if respondent_questions[movie_id].get('suggested', 0) == 1:
                    title = movie_mentions.get(movie_id, '')
                    if title:
                        ground_truth.append(title)
        
        # También revisar initiatorQuestions (por si el seeker acepta algo)
        initiator_questions = conversation.get('initiatorQuestions', {})
        for movie_id in mentioned_ids:
            if movie_id in initiator_questions:
                if initiator_questions[movie_id].get('suggested', 0) == 1:
                    title = movie_mentions.get(movie_id, '')
                    if title and title not in ground_truth:
                        ground_truth.append(title)
        
        return ground_truth
    
    def prepare_conversation_context(self, conversation: Dict, 
                                     context_ratio: float = 0.7) -> Tuple[str, List[str]]:
        """
        Extrae el contexto de una conversación para dar al LLM
        
        Args:
            conversation: diccionario con la conversación
            context_ratio: qué porcentaje de mensajes usar como contexto
            
        Returns:
            context_text: string con el contexto formateado
            ground_truth: lista de títulos de películas que deberían recomendarse
        """
        messages = conversation['messages']
        
        # Calcular cuántos mensajes usar como contexto
        n_context_messages = max(1, int(len(messages) * context_ratio))
        
        # Formatear el contexto
        context_text = self.format_conversation_context(
            conversation,
            n_context_messages
        )
        
        # Extraer ground truth de los mensajes restantes
        ground_truth = self.extract_ground_truth(conversation, n_context_messages)
        
        return context_text, ground_truth
    
    def get_evaluation_samples(self, n_samples: int = None, 
                              context_ratio: float = 0.7,
                              min_ground_truth: int = 1) -> List[Dict]:
        """
        Retorna muestras para evaluación
        
        Args:
            n_samples: cuántas muestras retornar (None = todas)
            context_ratio: proporción de la conversación a usar como contexto
            min_ground_truth: mínimo de películas en ground truth para incluir
            
        Returns:
            list de dicts: [{'context': str, 'ground_truth': list, 'conv_id': str}, ...]
        """
        samples = []
        
        conversations = self.conversations[:n_samples] if n_samples else self.conversations
        
        for conv in conversations:
            context, ground_truth = self.prepare_conversation_context(
                conv, 
                context_ratio
            )
            
            # Solo incluir si hay suficiente ground truth
            if len(ground_truth) >= min_ground_truth:
                samples.append({
                    'context': context,
                    'ground_truth': ground_truth,
                    'conv_id': conv['conversationId'],
                    'all_movies': list(conv['movieMentions'].values()),
                    'n_messages_total': len(conv['messages']),
                    'n_messages_context': int(len(conv['messages']) * context_ratio)
                })
        
        print(f"Prepared {len(samples)} samples with ground truth (min={min_ground_truth})")
        return samples

# Test mejorado
if __name__ == "__main__":
    dataset = ReDiALDataset('.', split='test')
    samples = dataset.get_evaluation_samples(n_samples=5)
    
    print("\n" + "="*80)
    print("SAMPLE 1 - DETAILED")
    print("="*80)
    print(f"Conversation ID: {samples[0]['conv_id']}")
    print(f"Total messages: {samples[0]['n_messages_total']}")
    print(f"Context messages: {samples[0]['n_messages_context']}")
    print("\nContext:")
    print(samples[0]['context'])
    print("\nGround Truth:", samples[0]['ground_truth'])
    print("\nAll movies mentioned:", samples[0]['all_movies'])
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total samples: {len(samples)}")
    gt_lengths = [len(s['ground_truth']) for s in samples]
    print(f"Avg ground truth size: {sum(gt_lengths)/len(gt_lengths):.2f}")
    print(f"Min/Max ground truth: {min(gt_lengths)}/{max(gt_lengths)}")