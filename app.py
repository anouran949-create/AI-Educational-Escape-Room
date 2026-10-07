import os, time, tempfile
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

st.set_page_config(page_title="AI Escape Room", page_icon="🔐")
st.title("🔐 AI Educational Escape Room")

# ---------- API key ----------
try:
    key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    key = st.sidebar.text_input("Google API Key", type="password")
if not key:
    st.info("Enter your Google API key in the sidebar to start.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = key

# ---------- Models ----------
class Puzzle(BaseModel):
    title: str = Field(description="Short title of the puzzle room")
    scene: str = Field(description="2-3 sentences of story setting the player is trapped in")
    puzzle_type: str = Field(description="cipher, reasoning, scenario, or cause-and-effect")
    question: str = Field(description="The puzzle the student must solve, requires understanding not memorizing")
    answer: str = Field(description="The correct solution")
    key_concept: str = Field(description="The curriculum concept this tests")
    difficulty: int = Field(description="1 to 5")

class GMReply(BaseModel):
    message: str = Field(description="What the Game Master says, in character")
    is_solved: bool = Field(description="True only if the student's message captures the key idea of the correct answer")
    hint_level: int = Field(description="0 if no hint was given, otherwise 1 to 3")

class Assessment(BaseModel):
    understanding_score: int = Field(description="0 to 100")
    attempts: int = Field(description="Number of answer attempts")
    hints_used: int = Field(description="Highest hint level needed, 0 to 3")
    strengths: list[str] = Field(description="What the student understood well")
    gaps: list[str] = Field(description="Misconceptions or missing knowledge")
    feedback: str = Field(description="2-3 encouraging sentences to the student")
    review_topics: list[str] = Field(description="Topics to revisit")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7, timeout=60, max_retries=1)

# ---------- Chains ----------
p_parser = PydanticOutputParser(pydantic_object=Puzzle)
g_parser = PydanticOutputParser(pydantic_object=GMReply)
a_parser = PydanticOutputParser(pydantic_object=Assessment)

puzzle_prompt = ChatPromptTemplate.from_template(
    """You are the puzzle designer of an educational escape room.
Use ONLY the curriculum text below. Do not use multiple choice.
Write the puzzle in {language}.

Curriculum text:
{context}

Topic: {topic}
Difficulty: {difficulty}

{format_instructions}""").partial(format_instructions=p_parser.get_format_instructions())

gm_prompt = ChatPromptTemplate.from_template(
    """You are a dramatic Game Master character that fits the scene, speaking to a student trapped in an escape room.
Stay in character. Reply in the language the student writes in. Max 4 sentences.

Scene: {scene}
Puzzle: {question}
Correct answer (SECRET, never reveal it): {answer}
Hints already given: {hints_used}

Conversation so far:
{history}

Student says: {student_message}

Rules:
- If the student's answer expresses the same key idea as the correct answer (exact wording is NOT required), set is_solved to true and celebrate.
- If the answer is wrong or partial, never reveal the correct answer. Encourage and nudge.
- If the student asks for a hint, give hint level {next_hint}:
  1 = vague pointer to the topic, 2 = narrower clue, 3 = strong clue but NOT the answer.
- Set hint_level to 0 when you gave no hint.

{format_instructions}""").partial(format_instructions=g_parser.get_format_instructions())

assess_prompt = ChatPromptTemplate.from_template(
    """You are an educational assessor. Evaluate the student's understanding
based on HOW they interacted with the puzzle, not only whether they solved it.

Concept tested: {key_concept}
Puzzle: {question}
Correct answer: {answer}
Solved: {solved}
Hints used (max level): {hints_used}

Full conversation:
{history}

Fewer hints and fewer wrong attempts mean stronger understanding.
Identify specific misconceptions from the wrong answers.
Write the feedback in {language}.

{format_instructions}""").partial(format_instructions=a_parser.get_format_instructions())

# ---------- State ----------
S = st.session_state
for k, v in {"retriever": None, "puzzle": None, "history": [], "hints": 0,
             "solved": False, "report": None, "language": "English"}.items():
    S.setdefault(k, v)

# ---------- Sidebar: teacher uploads curriculum ----------
st.sidebar.header("1) Curriculum")
up = st.sidebar.file_uploader("Upload PDF", type="pdf")
if up and st.sidebar.button("Index curriculum"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(up.read())
        path = f.name
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(PyPDFLoader(path).load())
    db = Chroma(embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
                collection_name=f"c{int(time.time())}")
    bar = st.sidebar.progress(0.0, "Indexing (free tier is slow, please wait)...")
    for i in range(0, len(chunks), 80):
        db.add_documents(chunks[i:i + 80])
        bar.progress(min(i + 80, len(chunks)) / len(chunks))
        if i + 80 < len(chunks):
            time.sleep(65)
    S.retriever = db.as_retriever(search_kwargs={"k": 4})
    st.sidebar.success(f"Done: {len(chunks)} chunks")

if S.retriever is None:
    st.info("Upload a curriculum PDF and click 'Index curriculum'.")
    st.stop()

# ---------- New puzzle ----------
st.sidebar.header("2) Puzzle")
topic = st.sidebar.text_input("Topic", "acid rain effects")
S.language = st.sidebar.selectbox("Language", ["English", "Arabic"])
difficulty = st.sidebar.slider("Difficulty", 1, 5, 2)
if st.sidebar.button("New puzzle"):
    docs = S.retriever.invoke(topic)
    context = "\n\n".join(d.page_content for d in docs)
    S.puzzle = (puzzle_prompt | get_llm() | p_parser).invoke(
        {"context": context, "topic": topic, "difficulty": difficulty, "language": S.language})
    S.history, S.hints, S.solved, S.report = [], 0, False, None

if S.puzzle is None:
    st.info("Choose a topic and click 'New puzzle'.")
    st.stop()

P = S.puzzle
st.subheader(P.title)
st.write("🔒 " + P.scene)
st.write("❓ **" + P.question + "**")

for h in S.history:
    with st.chat_message(h["role"]):
        st.write(h["text"])

# ---------- Chat ----------
if not S.solved:
    msg = st.chat_input("Your answer, or ask for a hint...")
    if msg:
        hist = "\n".join(f'{h["role"]}: {h["text"]}' for h in S.history)
        r = (gm_prompt | get_llm() | g_parser).invoke({
            "scene": P.scene, "question": P.question, "answer": P.answer,
            "hints_used": S.hints, "next_hint": min(S.hints + 1, 3),
            "history": hist, "student_message": msg})
        S.hints = max(S.hints, r.hint_level)
        S.history += [{"role": "user", "text": msg}, {"role": "assistant", "text": r.message}]
        S.solved = r.is_solved
        st.rerun()
else:
    st.success("🎉 The door is open!")

# ---------- Assessment ----------
if S.history and st.button("Finish and assess"):
    hist = "\n".join(f'{h["role"]}: {h["text"]}' for h in S.history)
    S.report = (assess_prompt | get_llm() | a_parser).invoke({
        "key_concept": P.key_concept, "question": P.question, "answer": P.answer,
        "solved": S.solved, "hints_used": S.hints, "history": hist, "language": S.language})

if S.report:
    R = S.report
    st.divider()
    st.metric("Understanding score", f"{R.understanding_score}/100")
    st.write(f"Attempts: {R.attempts} | Hints used: {R.hints_used}")
    st.write("**Strengths**"); [st.write("- " + x) for x in R.strengths]
    st.write("**Gaps**"); [st.write("- " + x) for x in R.gaps]
    st.write("**Feedback:** " + R.feedback)
    st.write("**Review topics**"); [st.write("- " + x) for x in R.review_topics]
