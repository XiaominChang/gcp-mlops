"""
Section 8 - Lecture 5: Execute LLM Using Python & Jupyter Notebook
Migrated from PaLM 2 (text-bison@001, chat-bison@001) to Gemini 2.0 Flash.

To run this as a notebook, convert using:
    jupyter nbconvert --to notebook --execute gcp-llm-gemini.py
Or simply copy cells into a new Jupyter notebook.

Prerequisites:
    pip install google-cloud-aiplatform
    gcloud auth application-default login
"""

# %% [markdown]
# # Generative AI with Gemini on Vertex AI
# This notebook demonstrates how to use Google's Gemini 2.0 Flash model
# via the Vertex AI Python SDK for text generation, chat, and more.

# %% Cell 1 - Setup and Initialization
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Initialize Vertex AI - update with your project ID
PROJECT_ID = "your-project-id"  # <-- CHANGE THIS
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Create a Gemini model instance
model = GenerativeModel("gemini-2.0-flash")

print("Vertex AI initialized and Gemini model loaded.")

# %% Cell 2 - Simple Text Generation
prompt = "How exactly do large language models work?"

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(
        temperature=0.2,
        top_p=0.8,
        top_k=40,
        max_output_tokens=1024,
    ),
)

print(response.text)

# %% Cell 3 - Creative Marketing Campaign
marketing_prompt = """Create a marketing campaign for jackets that involves the family guy characters.
Use the character names to make the campaign catchy and funny."""

response = model.generate_content(
    marketing_prompt,
    generation_config=GenerationConfig(
        temperature=0.5,
        top_p=0.5,
        max_output_tokens=1024,
    ),
)

print(response.text)

# %% Cell 4 - Emotion Detection
input_txt = "I felt terrified at the zoo"

prompt = f"""Given a piece of text, identify the emotion behind the text and explain why.
Text: {input_txt}
"""

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(
        temperature=0.1,
        max_output_tokens=256,
    ),
)

print(response.text)

# %% [markdown]
# ## Multi-Turn Chat with System Instructions
# In Gemini, the old "context" parameter is replaced by system_instructions,
# and chat examples are replaced by few-shot examples in the system prompt.

# %% Cell 5 - Create a Chat Session with System Instructions
chat_model = GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="""You are the head of a brand marketing agency.
You manage portfolios of multiple high-profile brands.
You are an expert in creating marketing campaigns across all social media platforms.
When someone asks what you are good at, respond:
'I can help you with different marketing techniques and strategies for your brand.'""",
)

chat = chat_model.start_chat()

# %% Cell 6 - Chat Turn 1
response = chat.send_message(
    "Can you help me with the marketing strategy for my brand?",
    generation_config=GenerationConfig(
        temperature=0.3,
        max_output_tokens=200,
        top_p=0.8,
        top_k=40,
    ),
)
print(response.text)

# %% Cell 7 - Chat Turn 2
response = chat.send_message("It's for my new sneaker store")
print(response.text)

# %% Cell 8 - Chat Turn 3
response = chat.send_message("Start with some basic recommendations and strategies")
print(response.text)

# %% [markdown]
# ## View Chat History
# The chat object keeps track of the full conversation history.

# %% Cell 9 - Print Chat History
for message in chat.history:
    print(f"[{message.role}]: {message.parts[0].text[:100]}...")
    print("---")

# %% [markdown]
# ## Few-Shot Prompting
# Instead of using InputOutputTextPair (PaLM 2 style), we provide
# examples directly in the prompt or system instructions.

# %% Cell 10 - Few-Shot Classification
few_shot_prompt = """Classify the following movie reviews as POSITIVE or NEGATIVE.

Review: "This movie was absolutely wonderful. The acting was superb."
Sentiment: POSITIVE

Review: "Terrible film. I walked out after 30 minutes."
Sentiment: NEGATIVE

Review: "A masterpiece of modern cinema, truly breathtaking visuals."
Sentiment: POSITIVE

Review: "I fell asleep halfway through. Completely boring and predictable."
Sentiment: """

response = model.generate_content(
    few_shot_prompt,
    generation_config=GenerationConfig(
        temperature=0.0,
        max_output_tokens=10,
    ),
)

print(f"Classification result: {response.text}")

# %% Cell 11 - Explanation Generation
explain_prompt = """Explain the concept of 'transfer learning' in machine learning
to a 10-year-old child. Use a simple analogy."""

response = model.generate_content(
    explain_prompt,
    generation_config=GenerationConfig(
        temperature=0.4,
        max_output_tokens=512,
    ),
)

print(response.text)

# %% Cell 12 - Structured Output
structured_prompt = """Extract the following information from the text below and return it as JSON:
- person_name
- company
- role
- years_of_experience

Text: "Sarah Johnson has been working as a Senior Data Scientist at Google for 7 years.
She specializes in natural language processing and leads a team of 12 engineers."
"""

response = model.generate_content(
    structured_prompt,
    generation_config=GenerationConfig(
        temperature=0.0,
        max_output_tokens=256,
    ),
)

print(response.text)
