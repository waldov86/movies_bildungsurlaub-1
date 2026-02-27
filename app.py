#%% Pakete
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import streamlit as st

#%% Ausgabeformat
class MovieOutput(BaseModel):
    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director of the movie")
    actors: list[str] = Field(description="The actors of the movie", examples=["Hanks, Tom", "Cruise, Tom"])
    release_year: int = Field(description="The release year of the movie")
    trailer_url: str = Field(description="The trailer url of the movie")
    probability: int = Field(description="The probability, that the user meant this film.", examples=[95, 88, 77])

class MoviesOutput(BaseModel):
    movies: list[MovieOutput]

parser = PydanticOutputParser(pydantic_object=MoviesOutput)
parser.get_format_instructions()


#%% Prompt Template
messages = [
    ("system", """Du bist ein Filmexperte und lieferst Informationen zu bestimmten Filmen.
    Der Nutzer übergibt dir die grobe Rahmenhandlung und du gibst in strukturierter Form die Informationen zurück.
    Gib die 3 kommerziell erfolgreichsten Filme zurück.
    Halte dich dabei strikt an das vorgegebene Schema {schema}
    """),
    ("user", "Rahmenhandlung: {handlung}")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages).partial(schema=parser.get_format_instructions())

#%% Modellinstanz erstellen
model = ChatGroq(
    model="openai/gpt-oss-120b",
)

#%% Chain
chain = prompt_template | model | parser


# %% Chain anwenden

prompt = st.chat_input(placeholder="Bitte beschreibe worum es in dem Film geht.")

# Prompt-Eingabe und Ergebnisverarbeitung
if prompt: 
    with st.chat_message("user"):   
        st.write(prompt)
    res = chain.invoke({"handlung": prompt})  # Prompt-Variable verwenden
    with st.chat_message("ai"):
        for r in res.movies:
            st.markdown(f"**Titel:** {r.title}")
            st.markdown(f"**Regisseur:** {r.director}")
            st.markdown(f"**Trailer-URL:** {r.trailer_url}")
            st.markdown(f"**Wahrscheinlichkeit:** {r.probability}")
# %%
