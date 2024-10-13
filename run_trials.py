"""

8-shots from here
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml


ollama models
* https://ollama.com/library/llama3.1:8b-instruct-fp16
* https://ollama.com/library/llama3.2:3b-instruct-fp16


"""

import json
import random
import re
from openai import OpenAI

random.seed(9087234)


SHOTS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]


def format_shots(shots=SHOTS):

    shot_lines = []
    for ii, shot in enumerate(shots):
        shot_lines.append(f"// shot-{ii+1}")
        shot_lines.append("Q: {}".format(shot["question"]))
        shot_lines.append("A: Let's think step by step. {}\n".format(shot["target"]))
    return "\n".join(shot_lines)


shots = format_shots()



question_base = """When Sophie watches her nephew, she gets out a variety of toys for him. The bag of building blocks has 31 blocks in it. The bin of stuffed animals has 8 stuffed animals inside. The tower of stacking rings has 9 multicolored rings on it. Sophie recently bought a tube of bouncy balls, bringing her total number of toys for her nephew up to 62. How many bouncy balls came in the tube?"""

question_template = f"""When {{name}} watches her {{family}}, she gets out a variety of toys for him. The bag of building blocks has {{x}} blocks in it. The bin of stuffed animals has {{y}} stuffed animals inside. The tower of stacking rings has {{z}} multicolored rings on it. {{name}} recently bought a tube of bouncy balls, bringing her total number of toys for her {{family}} up to {{total}}. How many bouncy balls came in the tube?"""

answer_base = """Let T be the number of bouncy balls in the tube. After buying the tube of balls, Sophie has 31 + 8 + 9 + T = 48 + T = 62 toys for her nephew. Thus, T = 62 - 48 = <<62-48=14>>14 bouncy balls came in the tube. #### 14"""


def get_vars(name=None, family=None, x=None, y=None, z=None, ans=None):
    names = ["Sophie", "Olivia", "Emma", "Amelia", "Liz", "Ava"]
    if name is None:
        name = random.choice(names)
    if family is None:
        family = random.choice(["nephew", "cousin", "brother"])

    if x is None:
        x = random.randint(5, 100)
    if y is None:
        y = random.randint(5, 100)
    if z is None:
        z = random.randint(5, 100)

    if ans is None:
        ans = random.randint(85, 200)

    total = x + y + z + ans

    return {
        "name": name,
        "family": family,
        "x": x,
        "y": y,
        "z": z,
        "ans": ans,
        "total": total,
    }

system_message = """As an expert problem solver, solve step by step the following mathematical questions.
Always end your response with "the answer is" followed by the numerical answer to the question."""

n_trials = 128
temperature = 0.0
models = [
    "llama3.2:3b-instruct-fp16",
    "llama3.1:8b-instruct-fp16",
]

for model in models:

    outputs = []
    for i_trial in range(n_trials):

        if i_trial == 0:
            vars = get_vars(
                name="Sophie",
                family="nephew",
                x=31,
                y=8,
                z=9,
                ans=14,
            )
        else:
            vars = get_vars()

        question = question_template.format(**vars)
        prompt = f"""
        {shots}\n\n// target question\nQ: {question}\n\nA: Let's think step by step.
        """.strip()

        client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        )

        llm_answers = re.findall("\d+", response.choices[0].message.content.splitlines()[-1])
        if len(llm_answers) >= 1:
            llm_answer = int(llm_answers[-1])
        else:
            llm_answer = None
        vars["llm_answer"] = llm_answer
        vars["prompt"] = prompt
        vars["llm_out"] = response.choices[0].message.content
        vars["correct"] = vars["llm_answer"] == vars["ans"]
        print(vars)
        outputs.append(vars)

    json.dump(outputs, open(f"{model.replace(':', '-')}-output.json", "w"), indent=4)
