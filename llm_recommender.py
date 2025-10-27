# llm_recommender.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import re

class LLMRecommender:
    """Wrapper genérico para cualquier LLM de HuggingFace"""
    
    def __init__(self, model_name: str, device: str = 'cuda', 
                 load_in_8bit: bool = False, max_memory_gb: int = 15):
        """
        Args:
            model_name: nombre del modelo en HuggingFace 
            device: 'cuda' o 'cpu'
            load_in_8bit: si usar cuantización 8-bit
            max_memory_gb: GB máximos a usar
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading {model_name}...")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configurar pad token si no existe
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo con cuantización si se solicita
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                max_memory={0: f"{max_memory_gb}GB"}
            )
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                max_memory={0: f"{max_memory_gb}GB"}
            )
        
        self.model.eval()  # modo evaluación
        print(f"Model loaded successfully on {self.device}!")
        print(f"Model size: ~{self.get_model_size_gb():.2f} GB")
    
    def get_model_size_gb(self) -> float:
        """Calcula el tamaño del modelo en GB"""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024**3)
    
    def generate_recommendations(self, context: str, prompt_template, 
                                max_new_tokens: int = 250,
                                temperature: float = 0.7) -> tuple:
        """
        Genera recomendaciones dado un contexto
        
        Args:
            context: string con el contexto de la conversación
            prompt_template: función que formatea el prompt
            max_new_tokens: máximo de tokens a generar
            temperature: temperatura para sampling
            
        Returns:
            response: string con la respuesta del modelo
            latency: tiempo de generación en segundos
        """
        # Formatear prompt
        prompt = prompt_template(context)
        
        # Tokenizar
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048  # limitar contexto para evitar OOM
        ).to(self.device)
        
        # Generar
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # evitar repetición
                top_p=0.9,  # nucleus sampling
            )
        
        latency = time.time() - start_time
        
        # Decodificar solo la parte nueva
        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        return response, latency
    
    def extract_movie_titles(self, response: str, max_recommendations: int = 10) -> list:
        """
        Extrae títulos de películas de la respuesta del modelo
        VERSIÓN MEJORADA: separa título de descripción
        
        Args:
            response: string con la respuesta del modelo
            max_recommendations: máximo de recomendaciones a retornar
            
        Returns:
            list de strings con títulos de películas (solo los títulos)
        """
        movies = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Saltar líneas vacías o muy cortas
            if len(line) < 3:
                continue
            
            # Saltar líneas que no parecen ser recomendaciones
            skip_phrases = ['here are', 'based on', 'you might', 'i recommend', 
                           'these are', 'recommendation', 'user:', 'assistant:', 
                           'given this', 'conversation']
            if any(phrase in line.lower()[:50] for phrase in skip_phrases):
                continue
            
            # Remover numeración común: "1.", "1)", "- ", etc.
            line = re.sub(r'^[\d\-\.\)\*\•\s]+', '', line)
            
            # NUEVO: Separar título de descripción
            # Buscar el primer separador: ":", "-", "–", o dos espacios
            separators = [':', ' -', ' –', '  ']
            title = line
            
            for sep in separators:
                if sep in line:
                    title = line.split(sep)[0].strip()
                    break
            
            # Remover comillas si las hay
            title = title.strip('"\'')
            
            # Limpiar caracteres extraños al final
            title = re.sub(r'[:\-–,\.]$', '', title).strip()
            
            # Filtros de calidad
            if 3 < len(title) < 100:  # longitud razonable
                movies.append(title)
            
            if len(movies) >= max_recommendations:
                break
        
        return movies[:max_recommendations]
    
    def clear_memory(self):
        """Limpia memoria GPU"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()

# Test mejorado
if __name__ == "__main__":
    llm = LLMRecommender("google/gemma-3-1b-it", load_in_8bit=False)
    
    test_context = """User: I love action movies with great special effects like "The Matrix (1999)"
Recommender: What kind of action movies do you prefer?
User: Sci-fi action with deep philosophical themes."""
    
    # Prompt más restrictivo
    def simple_prompt(context):
        return f"""Given this conversation about movies, recommend exactly 5 movies.

Conversation:
{context}

Output ONLY a numbered list of movie titles with years. No descriptions or explanations.

Recommendations:
1."""
    
    print("Testing LLM...")
    response, latency = llm.generate_recommendations(
        test_context, 
        simple_prompt,
        max_new_tokens=150,  # menos tokens
        temperature=0.3  # más determinístico
    )
    
    print(f"\n{'='*80}")
    print(f"Raw Response ({latency:.2f}s):")
    print(f"{'='*80}")
    print(response)
    
    print(f"\n{'='*80}")
    print("Extracted Movies:")
    print(f"{'='*80}")
    movies = llm.extract_movie_titles(response)
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie}")
    
    # Test de extracción con texto complejo
    print(f"\n{'='*80}")
    print("Testing extraction with descriptions:")
    print(f"{'='*80}")
    
    test_response = """1. Blade Runner 2049: A visually stunning sequel
2. Inception (2010) - Mind-bending thriller
3. The Matrix Reloaded
4.  Arrival  –  Sci-fi drama about communication
5. Interstellar (2014)"""
    
    extracted = llm.extract_movie_titles(test_response)
    print("Input:")
    print(test_response)
    print("\nExtracted:")
    for i, movie in enumerate(extracted, 1):
        print(f"{i}. {movie}")
    
    llm.clear_memory()