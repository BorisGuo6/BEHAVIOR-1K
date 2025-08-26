
fwd_prompt = """
You are a capable agent designed to infer forward dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the current state of the scene.
- The subsequent four images (A, B, C, D) represent potential next states based on the given state changes.

Your task is to identify which choice (A, B, C, or D) is the most likely next state after the following state changes take place:

{STATE_CHANGES}

Focus on the visual details in the images, specifically observing the positions, placements, and relationships of the objects mentioned.

Do not perform any reasoning or explanation; simply select the letter corresponding to the most likely next state.
"""

inv_prompt = """You are a capable agent designed to infer inverse dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the **current state** of the scene.
- The second image represents the **next state** of the scene.

Your task is to determine which set of state changes listed below(A, B, C, or D) most accurately describes the action(s) that occurred between the two states. Each choice contains a list of state changes that describe object interactions and positional changes:


{STATE_CHANGES_CHOICES}
Focus on the visual details in the images, such as the relationships, placements, and interactions of the objects mentioned in the choices. Use this information to identify the correct sequence of state changes. Do not perform any reasoning or explanation; simply select the letter corresponding to the correct set of state changes, and only generate a single capitalized letter as your output.
"""

multi_fwd_prompt = """
You are a capable agent designed to infer multi-step forward dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the **current state** of the scene.
- The subsequent four options (A, B, C, D) are presented as **sequences of images** (filmstrips), each representing a potential sequence of future states.

Your task is to identify which choice (A, B, C, or D) shows the most likely sequence of states after the following actions take place **in order**:

{STATE_CHANGES}

Focus on the visual details in the images, specifically observing the positions, placements, and relationships of the objects and how they evolve through **each step** of the sequence.

Do not perform any reasoning or explanation; simply select the letter corresponding to the most likely sequence of states.
"""

multi_inv_prompt = """You are a capable agent designed to infer **multi-step inverse dynamics** transitions in embodied decision-making. Analyze the provided **sequence of images (filmstrip)**, which represents the evolution of a scene over multiple steps.

- The filmstrip shows the **entire sequence of states**, from the initial state to the final state.

Your task is to determine which set of **ordered state changes** listed below (A, B, C, or D) most accurately describes the full sequence of actions that occurred to transition through all the states shown. Each choice contains a numbered list of state changes corresponding to each step in the sequence:


{STATE_CHANGES_CHOICES}
Focus on the visual details in the images, paying close attention to how the relationships, placements, and interactions of objects change from one frame to the next. Use this information to identify the correct **sequence** of state changes. Do not perform any reasoning or explanation; simply select the letter corresponding to the correct set of state changes, and only generate a single capitalized letter as your output.
"""


multi_fwd_ordering_prompt = '''You are a capable agent designed to infer multi-step forward dynamics transitions in embodied decision-making. Your goal is to predict the correct sequence of future states that result from applying a given series of actions to an initial state.

## Your Task
You will be provided with a single **Current State Image** and a set of shuffled **Future State Images** (labeled 1, 2, 3, etc.). To determine their correct order, you must follow the sequence of actions provided below.

1.  Start with the **Current State Image**.
2.  Apply the **first action** from the `Actions in Order` list to this state.
3.  Find the **Future State Image** that matches the outcome of this action. This is the first state in the correct sequence.
4.  Next, apply the **second action** to the state you just identified.
5.  Find the corresponding image among the remaining future states.
6.  Continue this process until all actions have been applied and all future states have been ordered.

## Output Format
Your response **must be only** a Python list of integers representing the correct chronological order of the future state image labels. Do not include any other text, reasoning, or explanation.

**Example:** If you determine the correct sequence is 'Next State 1' -> 'Next State 3' -> 'Next State 2', your output must be:
`[1, 3, 2]`

## Actions in Order
{STATE_CHANGES}

Now please provide your answer in the requested format.
'''

multi_inv_ordering_prompt = '''You are a capable agent designed to infer multi-step inverse dynamics transitions in embodied decision-making. Your goal is to determine the correct chronological order of actions that caused the state transitions shown in a sequence of images.

## Your Task
You will be given an ordered sequence of images that show a scene evolving over time, along with a shuffled list of the actions that caused these changes. To solve this, you must:
1.  Analyze the transition from the first image to the second. Determine the specific visual change that occurred.
2.  From the **Shuffled Actions** list provided below, identify the single action that best describes this change.
3.  Repeat this process for all subsequent pairs of images (second to third, third to fourth, etc.) until you have correctly ordered all the actions.

## Output Format
Your response **must be only** a Python list of integers representing the correct order of the action labels. Do not include any other text, reasoning, explanations, or code formatting.

**Example:** If the correct sequence is [Action 2] -> [Action 3] -> [Action 1], your output must be:
`[2, 3, 1]`

## Shuffled Actions
{SHUFFLED_ACTIONS}

Now please provide your answer in the requested format.
'''