#pip install langchain-community
#pip install langchain
#pip install pypdf
#pip install langchain-chroma
#pip install bitsandbytes
#pip install transformers
#pip install sentence-transformers

import torch
from typing import List
from tkinter import *
from tkinter.filedialog import askopenfilename
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

READER_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" #"microsoft/Phi-3-mini-4k-instruct" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
FILE_PATH = "..."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
QUESTION_TO_ASK = ""
QUANTIFICATION_4BITS = True

def load_reader_llm(model_name: str):
    if QUANTIFICATION_4BITS:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reader_llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
    return reader_llm, tokenizer


def split_documents(chunk_size: int, chunk_overlap: int, knowledge_base: List, tokenizer_name: str) -> List:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(knowledge_base)

class Interface(Tk):
    purple = "#DEABED"
    green = "#C9FFCB"
    red = "#F8CEC2"
    bg = "white"
    def __init__(self):
        Tk.__init__(self)
        self.title('RAG')
        bg_par = type(self).bg
        l_par = {"READER_MODEL_NAME":READER_MODEL_NAME,"Question":QUESTION_TO_ASK,
                 "EMBEDDING_MODEL_NAME":EMBEDDING_MODEL_NAME,"FILE_PATH":FILE_PATH 
                 ,"CHUNK_SIZE":CHUNK_SIZE
                 , "CHUNK_OVERLAP":CHUNK_OVERLAP,
                 "QUANTIFICATION_4BITS":str(QUANTIFICATION_4BITS)}
        for i in l_par.keys():
            l_par[i] = StringVar(self,l_par[i])
        paned_main = PanedWindow(self,bg=bg_par)
        paned_parametres = PanedWindow(self, orient='vertical',bg=bg_par)
        paned_main.add(paned_parametres)
        for i in l_par.keys():
            if i in ['Question']:
                continue
            lttle_paned = PanedWindow(self,orient='horizontal', width=500,bg='white')
            lttle_paned.add(Label(self,text=f"{i} \t:",bg=bg_par))
            lttle_paned.add(Entry(self,text=l_par[i]))
            if i == 'FILE_PATH':
                lttle_paned.add(Button(self,text="Parcourir",command=self.parcourir))
            paned_parametres.add(lttle_paned)
        #paned_load = PanedWindow(self,orient='vertical',bg=bg_par,width=500)
        paned_load = paned_parametres
        paned_main.add(paned_load)
        paned_quest = PanedWindow(self,orient='horizontal',bg=bg_par)
        paned_quest.add(Label(self,text="Poser une question : ",bg=bg_par))
        paned_quest.add(Entry(self,text=l_par['Question']))
        paned_load.add(paned_quest)
        paned_load.add(Button(self,text='Poser la question', command=self.load_main))
        self.step = {}
        self.step[5] = Label(self,text='Device : ?',bg='white')
        self.step[0] = Label(self,text='Chargement des poids du modèle',bg='cyan')
        self.step[1] = Label(self,text='Découpage du documents en chunks',bg='cyan')
        self.step[2] = Label(self,text='Chargement du modèle d\'embedding',bg='cyan')
        self.step[3] = Label(self,text='Extraction des informations pertinentes du documents ', bg='cyan')
        self.step[4] = Label(self,text='Élaboration du prompt', bg='cyan')
        self.step[6] = Label(self,text='Génération d\'une réponse', bg='cyan')
        for i in self.step.keys():
            self.step[i]['justify'] = 'left'
            paned_load.add(self.step[i])
        paned_answer = PanedWindow(self,orient='vertical',width=700,bg=bg_par,height=800)
        paned_answer.add(Label(self,text='Réponse : ',justify="left",bg=bg_par))
        self.label_reponse = Label(self,text='',wraplength=600,bg=bg_par,justify="left")
        paned_answer.add(self.label_reponse)
        paned_main.add(paned_answer)
        paned_main.pack()
        self.l_par = l_par
    def parcourir(self):
        self.str_parcours = askopenfilename()
        if self.str_parcours:
            self.l_par["FILE_PATH"].set(self.str_parcours)
    def check_par(self):
        global READER_MODEL_NAME, QUESTION_TO_ASK,QUANTIFICATION_4BITS, FILE_PATH, CHUNK_SIZE, EMBEDDING_MODEL_NAME, CHUNK_OVERLAP
        READER_MODEL_NAME = self.l_par['READER_MODEL_NAME'].get()
        QUESTION_TO_ASK = self.l_par['Question'].get()
        FILE_PATH = self.l_par['FILE_PATH'].get()
        EMBEDDING_MODEL_NAME = self.l_par['EMBEDDING_MODEL_NAME'].get()
        CHUNK_SIZE = int(self.l_par['CHUNK_SIZE'].get())
        CHUNK_OVERLAP = int(self.l_par['CHUNK_OVERLAP'].get())
        QUANTIFICATION_4BITS = self.l_par['QUANTIFICATION_4BITS'].get().strip().lower() == "true"
        self.step[0]['bg'] = type(self).purple
        self.after(5,self.load_llm)
    def load_main(self):
        self.step[5]['text'] = f'Device : {DEVICE}'
        for i in self.step.keys():
            if self.step[i]['bg'] == type(self).bg:
                continue
            self.step[i]['bg'] = type(self).red
        self.after(5,self.check_par)
    def load_llm(self):
        self.reader_llm, self.reader_tokenizer = load_reader_llm(READER_MODEL_NAME)
        self.step[0]['bg'] = type(self).green
        self.step[1]['bg'] = type(self).purple
        self.after(5,self.load_docs)
    def load_docs(self):
        loader = PyPDFLoader(FILE_PATH)
        raw_knowledge_base = loader.load()
        self.docs_processed = split_documents(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            knowledge_base=raw_knowledge_base,
            tokenizer_name=EMBEDDING_MODEL_NAME,
        )
        self.step[1]['bg'] = type(self).green
        self.step[2]['bg'] = type(self).purple
        self.after(5,self.embed_model)
    def embed_model(self):
        #print('\nEmbedding model\n')
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=False,  
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.step[2]['bg'] = type(self).green
        self.step[3]['bg'] = type(self).purple
        self.after(5,self.transformation_)
    def transformation_(self):
        #print('\nVectorisation du documents\n')
        vector_store = Chroma.from_documents(
            self.docs_processed,
            self.embedding_model,
            persist_directory=f"db_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
            collection_metadata={"hnsw:space": "cosine"},
        )
        #print('\nResults\n')

        self.results = vector_store.similarity_search_by_vector(
            embedding=self.embedding_model.embed_query(QUESTION_TO_ASK),
            k=4,
        )
        
        self.step[3]['bg'] = type(self).green
        self.step[4]['bg'] = type(self).purple
        self.after(5,self.elabor)
    def elabor(self):
        #print('\nGénération du prompt\n')

        retrieved_docs_text = [doc.page_content for doc in self.results]
        context = "\nExtracted documents:\n\n"
        context += "\n\n".join(
            f"Document {i}:::\n{doc}" for i, doc in enumerate(retrieved_docs_text)
        )

        prompt_in_chat_format = [
            {
                "role": "system",
                "content": (
                    "Using the information contained in the context, give a comprehensive "
                    "answer to the question. Respond only to the question asked, response "
                    "should be concise and relevant to the question. If the answer cannot "
                    "be deduced from the context, do not give an answer."
                ),
            },
            {
                "role": "user",
                "content": f"""Context: {context}
    ---
    Now here is the question you need to answer.
    Question: {QUESTION_TO_ASK}""",
            },
        ]
        
        self.final_prompt = self.reader_tokenizer.apply_chat_template(
            prompt_in_chat_format,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.step[4]['bg'] = type(self).green
        self.step[6]['bg'] = type(self).purple
        self.after(5,self.final_answer)
    def final_answer(self):
        #print('\nFinal Answer\n')
        answer = self.reader_llm(self.final_prompt)[0]["generated_text"]
        print('\nRéponse générée : \n')
        print(answer)
        print('\n \n')
        self.label_reponse['text'] = answer
        self.step[6]['bg'] = type(self).green


if __name__ == "__main__":
    i = Interface()
    i.mainloop()
