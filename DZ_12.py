import amrlib
from transformers import pipeline

model_directory = '/Users/sydorukoksana/newapp/.venv/lib/python3.10/site-packages/amrlib/data/model_parse_xfm_bart_large-v0_1_0 2'

try:
    stog_model = amrlib.load_stog_model(model_directory)
except FileNotFoundError:
    print("STOG model not found. Please ensure that it is downloaded and located in the correct directory.")
    exit(1)

print("Available methods and attributes in stog_model:")
print(dir(stog_model))

def get_amr_graph(sentence):
    try:
        amr_graphs = stog_model.parse_sents([sentence])
        return amr_graphs[0] if amr_graphs else None
    except Exception as e:
        print(f"Error generating AMR graph: {e}")
        return None

summarizer = pipeline("summarization")

def summarize_text(text, max_length=50, min_length=25):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

sentence = "The dog barked loudly at the stranger."
amr_graph = get_amr_graph(sentence)
print("AMR Graph for sentence:")
print(amr_graph)

text = """
Climate change is a significant global challenge that affects ecosystems, weather patterns, and sea levels. 
It results from human activities such as burning fossil fuels, deforestation, and industrial processes. 
To combat climate change, nations around the world are working towards reducing greenhouse gas emissions 
and transitioning to renewable energy sources.
"""

summary = summarize_text(text)
print("\nSummary of text:")
print(summary)

"""
AMR Graph for sentence:
# ::snt The dog barked loudly at the stranger.
(b / bark-01
      :ARG0 (d / dog)
      :ARG2 (s / stranger)
      :manner (l / loud))

Summary of text:
 Climate change is a significant global challenge that affects ecosystems, weather patterns, and sea levels . It results from burning fossil fuels, deforestation, and industrial processes . Nations around the world are working towards reducing greenhouse gas emissions .
"""
