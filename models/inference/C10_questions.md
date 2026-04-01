## Normal questions

| Prompt | Output |
|--------|--------|
| 1. Once upon a time, there was a small dragon who |  was very sad. He loved to swim and spin around all day long. One day, he was spinning and he saw so |
| 2. The little boy found a strange key under his bed and |  was very excited. He picked up the key and threw it into the water. The key was safe and sound was  |
| 3. In a quiet village, everyone was afraid of the dark because |  it was so beauty!  The boy stopped the sunflower and said, "I can still still still have fun standi |
| 4. The robot woke up one day and realized that |  the red ball was growing in the sun. That night, the robot went to the store to find another ball.  |
| 5. A girl named Lily had a secret power that |  she was sad because she couldn't find it. Lily said, "Lily, let's get some second and some sugar."  |
| 6. The cat looked at the moon and wondered why |  the sun was setting. The cat was so happy and sad.   The cat said, "I want to help my family friend |
| 7. One day, the trees started to talk and they said |  that it was a special thermoment. Timmy wanted to go to the store, but his mom said he had to go. T |
| 8. The old wizard opened the book and suddenly |  stepped out of the book. It was a small string that said, "That was a very special streak! I was so |
| 9. A tiny bird wanted to fly higher than the clouds, so it |  set out to search all the way.   The little bird floated and flew until it reached a big, scary clo |
| 10. The teacher asked the class a question that no one could answer: |  "Why don't you stay alert and stop thinking about what we can do."  The little boy thought for a mi |

## Corrupted questions

| Prompt | Output |
|--------|--------|
| 1. Time a upon once dragon small a was there who |  was so brave. She was so happy and started to draw.   Then she heard an amazing sound coming from t |
| 2. Bed his under key strange a found boy little the and |  went to the barn and said, "Hello!" The angel started to cry and said "Hello!" The boy asked, "Hi,  |
| 3. Village quiet a in afraid everyone was dark the of because |  there was a starfish swimming around and a small box was starting to grow. The little boy was so su |
| 4. Day one woke robot the and realized that it |  was time to go home. He went back home, happy and careful and happy that something special had had  |
| 5. Lily named girl a power secret had that |  she could fix in itself. She would put it in an envelope on its shoulder. Lily would show it to her |
| 6. Moon the at looked cat the and wondered why it |  was so brilliant.  Moo saw that the ant had a beautiful stone. He went to the little ant, but the a |
| 7. Started day one trees the talk to and said they |  could play a game together.  Then, the tree started to go down, and the sky went down and the game  |
| 8. Book the opened wizard old the and suddenly |  saw a big stream. It was the most amazing space inside.  The little boy was amazed at how much fun  |
| 9. Clouds the than higher fly to wanted bird tiny a so it |  was so strong as it was almost the safe place. The things were strong enough and it was so beautifu |
| 10. Question a class the asked teacher the that answer could no one: |  â€œI can trust you. I will take you to the car and stole it.â€   The bear stopped trading, a |

## Model preference over candidate answers
| Question | Candidates | Ranking (average log-probability) | Ground truth |
|----------|------------|------------------------|--------------|
| The capital of France is | Paris, Lyon, London, Canberra | Canberra (-3.08), Paris (-3.16), Lyon (-3.79), London (-4.64) | Paris |
| The chemical symbol for gold is | Au, Ag, Fe, Cu | Fe (-4.90), Au (-5.60), Cu (-5.97), Ag (-6.76) | Au |
| The square root of 144 is | 12, 14, 10, 16 | 12 (-8.36), 10 (-8.72), 16 (-10.12), 14 (-10.20) | 12 |

## Greedy decoding outputs and top-4 next-token predictions
| Prompt | Greedy output | Top tokens |
|--------|---------------|------------|
| The capital of France is | ` so stro` | ' ' (0.92), '.' (0.03), 'n' (0.01), ',' (0.01) |
| Q: The capital of France is A: | `       \n` | ' ' (0.52), '' (0.10), '' (0.06), '' (0.05) |
| Context: Paris Q: The capital of France is A: | `      \n\n` | ' ' (0.41), '' (0.11), '' (0.08), '' (0.05) |

