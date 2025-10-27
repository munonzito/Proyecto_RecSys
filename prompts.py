def zero_shot_prompt(context):
    """Prompt básico zero-shot"""
    return f"""You are a movie recommendation expert. Given the following conversation between a user seeking movie recommendations and a recommender, suggest 10 movies that the user would enjoy based on their preferences.

Conversation:
{context}

Provide exactly 10 movie recommendations as a numbered list. Output ONLY the movie titles with their release year in parentheses, one per line, nothing else.

Recommendations:
1."""

def few_shot_prompt(context):
    """Prompt con ejemplos (few-shot)"""
    examples = """Here are some examples:

Example 1:
User: I love action movies with great special effects like "The Matrix (1999)"
Recommender: What aspects do you enjoy most?
User: The sci-fi elements and philosophical themes

Recommendations:
1. Inception (2010)
2. Blade Runner (1982)
3. The Prestige (2006)
4. Minority Report (2002)
5. Total Recall (1990)

Example 2:
User: I enjoy romantic comedies like "When Harry Met Sally (1989)"
Recommender: Do you prefer classic or modern rom-coms?
User: I like both, but prefer ones with witty dialogue

Recommendations:
1. Sleepless in Seattle (1993)
2. You've Got Mail (1998)
3. Notting Hill (1999)
4. The Proposal (2009)
5. Crazy, Stupid, Love (2011)

"""
    return f"""{examples}

Now provide recommendations for this conversation:

Conversation:
{context}

Provide exactly 10 movie recommendations as a numbered list. Output ONLY the movie titles with their release year in parentheses, one per line.

Recommendations:
1."""

def cot_prompt(context):
    """Chain-of-thought prompt"""
    return f"""You are a movie recommendation expert. Analyze the following conversation and recommend movies.

Conversation:
{context}

First, analyze what the user likes (genres, themes, styles). Then provide 10 movie recommendations.

Analysis of user preferences:
- Genre preferences: [identify genres mentioned]
- Specific movies liked: [list movies user responded positively to]
- Themes/styles: [identify patterns]

Based on this analysis, here are 10 movie recommendations:
1."""

def role_prompt(context):
    """Role-based prompt with specific persona"""
    return f"""You are an experienced movie critic and recommendation specialist with deep knowledge of cinema history, genres, and audience preferences. A user is seeking movie recommendations based on this conversation:

Conversation:
{context}

As an expert recommender, provide 10 carefully selected movie recommendations that match the user's taste. Consider genre, themes, era, and style.

Your 10 recommendations:
1."""

def structured_prompt(context):
    """Prompt pidiendo output estructurado"""
    return f"""Given this movie conversation, provide 10 movie recommendations.

Conversation:
{context}

Output format - provide exactly 10 movies in this format:
1. [Movie Title] ([Year])
2. [Movie Title] ([Year])
...

Recommendations:
1."""

# Diccionario para fácil acceso
PROMPT_STRATEGIES = {
    'zero_shot': zero_shot_prompt,
    'few_shot': few_shot_prompt,
    'chain_of_thought': cot_prompt,
    'role_based': role_prompt,
    'structured': structured_prompt
}

# Test
if __name__ == "__main__":
    test_context = """User: I love action movies with great special effects like "The Matrix (1999)"
Recommender: What kind of action movies do you prefer?
User: Sci-fi action with deep philosophical themes."""
    
    print("Testing prompts:\n")
    for name, prompt_fn in PROMPT_STRATEGIES.items():
        print(f"=== {name.upper()} ===")
        print(prompt_fn(test_context)[:300] + "...")
        print()