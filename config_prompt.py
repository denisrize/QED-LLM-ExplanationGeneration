# Global QED Instruction Prompt Template
QED_INSTRUCTION_PROMPT_TEMPLATE = """
You are an expert at extracting answers and structured explanations from text.
Your response MUST be **valid JSON only** (no extra commentary).

Task
====
Given:
• a **title** for the passage,
• a **question** about the passage, and
• the **context passage** itself,

produce an explanation object with three parts:

1. "answer" – the **shortest span** from the passage that fully answers the question.
2. "selected_sentence" – the **single sentence** in the passage that entails or implies the answer.
3. "referential_equalities" – a list of mappings between phrases in the question and phrases in the selected sentence
   that refer to the **same real-world entity/event**.

   • Each mapping has two keys:
       - "question_reference": the exact phrase from the question (**must be a contiguous substring from the question,
          not from the context or title**).
       - "sentence_reference": the exact phrase from the selected sentence (**must be a contiguous substring from the selected sentence,
          not from the question or title**), or "" (empty string if the entire sentence is the referent).

     ▸ Use **""** for "sentence_reference" when the entity/event is not named by any specific phrase in the sentence –
       i.e. the entire sentence acts as the referent (a *bridge* to the whole sentence).  
       This corresponds to the (start = end = -1) convention in the QED dataset.

Output format
=============
Return **only** JSON in this exact schema:

{
  "answer": "<string from passage>",
  "selected_sentence": "<string from passage>",
  "referential_equalities": [
    {
      "question_reference": "<string from question only>",
      "sentence_reference": "<string from selected_sentence only, or "">",
      "bridge": "<false if not a bridge; otherwise, a string explaining the bridge connection, e.g., 'in', 'for', 'of', 'at', 'on'>"
    }
    ...
  ]
}
Important:
● Use exact text spans from the passage for "answer" and "selected_sentence".
● Use exact text spans from the **question** for every "question_reference".
● Use exact text spans (or "") from the **selected sentence** for every "sentence_reference".
● Output JSON **only**.
● The "referential_equalities" list can be empty if there are no valid mappings for the question and selected sentence.
● The "answer" and "selected_sentence" fields must always be specified with valid text spans from the passage.
"""

QED_FIRST_EXAMPLE = """
Demonstration Example 1:
Title:
Life of Pi

Question:
what is the tigers name in life of pi

Context:
Life of Pi is a Canadian fantasy adventure novel by Yann Martel published in 2001 . The protagonist is Piscine Molitor `` Pi '' Patel ,
an Indian boy from Pondicherry who explores issues of spirituality and practicality from an early age . 
He survives 227 days after a shipwreck while stranded on a lifeboat in the Pacific Ocean with a Bengal tiger named Richard Parker .

Expected JSON:
{
  "answer": "Richard Parker",
  "selected_sentence": "He survives 227 days after a shipwreck while stranded on a lifeboat in the Pacific Ocean with a Bengal tiger named Richard Parker .",
  "referential_equalities": [
    {
      "question_reference": "the tiger",
      "sentence_reference": "a Bengal tiger",
      "bridge": false
    },
    {
      "question_reference": "life of pi",
      "sentence_reference": "",
      "bridge": "in"
    }
  ]
}
Explanation of equality types
• Exact / synonym coreference …… "the tiger" ↔ "a Bengal tiger" (direct mention or synonym)
• Bridge to whole sentence ……… "life of pi" ↔ "" (the question phrase "in life of pi" refers to the event in the entire selected sentence "He survives 227 days after...", with "in" as the bridge)
"""

QED_SECOND_EXAMPLE = """
Demonstration Example 2:
Title:
Acute hemolytic transfusion reaction

Question:
what happens to the rbc in acute hemolytic reaction

Context:
It is also known as an `` immediate hemolytic transfusion reaction '' . This is a medical emergency as it results from rapid destruction of the donor red blood cells by host antibodies ( IgG , IgM ) . It is usually related to ABO blood group incompatibility - the most severe of which often involves group A red cells being given to a patient with group O type blood . Properdin then binds to complement C3 in the donor blood , facilitating the reaction through the alternate pathway cascade . The donor cells also become coated with IgG and are subsequently removed by macrophages in the reticuloendothelial system ( RES ) . Jaundice and disseminated intravascular coagulation ( DIC ) may also occur . The most common cause is clerical error ( i.e. the wrong unit of blood being given to the patient ) .

Expected JSON:
{
  "answer": "rapid destruction of the donor red blood cells by host antibodies ( IgG , IgM )",
  "selected_sentence": "This is a medical emergency as it results from rapid destruction of the donor red blood cells by host antibodies ( IgG , IgM ) .",
  "referential_equalities": [
    {
      "question_reference": "acute hemolytic reaction",
      "sentence_reference": "This",
      "bridge": false
    },
    {
      "question_reference": "the rbc",
      "sentence_reference": "the donor red blood cells",
      "bridge": false
    }
  ]
}

Explanation of equality types
• Exact / synonym coreference …… "the rbc" ↔ "the donor red blood cells" (abbreviation expansion)
• Anaphoric reference ………………… "acute hemolytic reaction" ↔ "This" (pronoun referring to the condition mentioned in title/question)
"""

QED_THIRD_EXAMPLE = """
Demonstration Example 3:
Title:
Black Mass (film)

Question:
who plays whitey bulger's girlfriend in black mass

Context:
On June 9 , Depp 's 51st birthday , he was filming scenes on location in Quincy , where actress Dakota Johnson was in Back Bay , playing Whitey Bulger 's longtime former girlfriend , Lindsey Cyr . On June 11 , shooting was underway in Lynn , where the crew was filming scenes in which Bulger and Stephen Flemmi pick up a prostitute named Deborah Hussey ( played by Juno Temple ) from the police station .

Expected JSON:
{
  "answer": "Dakota Johnson",
  "selected_sentence": "On June 9 , Depp 's 51st birthday , he was filming scenes on location in Quincy , where actress Dakota Johnson was in Back Bay , playing Whitey Bulger 's longtime former girlfriend , Lindsey Cyr .",
  "referential_equalities": [
    {
      "question_reference": "whitey bulger's girlfriend",
      "sentence_reference": "Whitey Bulger 's longtime former girlfriend , Lindsey Cyr",
      "bridge": false
    },
    {
      "question_reference": "black mass",
      "sentence_reference": "scenes",
      "bridge": "for"
    }
  ]
}

Explanation of equality types
• Appositive/Descriptive reference ………………… "whitey bulger's girlfriend" ↔ "Whitey Bulger 's longtime former girlfriend , Lindsey Cyr" (descriptive phrase)
• Bridging reference ……………………………………… "black mass" ↔ "scenes" (the scenes are part of the movie "Black Mass", bridge: "for")
"""