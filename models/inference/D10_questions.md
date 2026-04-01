## Normal questions

| Prompt | Output |
|--------|--------|
| 1. Once upon a time, there was a small dragon who |  was very sad. He loved to play outside and explore the world around his house. One day he saw a big |
| 2. The little boy found a strange key under his bed and |  wanted to play it with.  He started to crawl around the garden. He saw a big box of balls and the s |
| 3. In a quiet village, everyone was afraid of the dark because |  he was so happy! He had never heard an extra beautiful sound after the dark.   The beautiful sound  |
| 4. The robot woke up one day and realized that |  the ribbon's favorite things wanted to do if the ribbon's heed would leave, but it wasn't so gratef |
| 5. A girl named Lily had a secret power that |  she loved to sing and dance. They were very proud of their power of papers and singing. One day, th |
| 6. The cat looked at the moon and wondered why |  he was so envious.  "Why is that moon talled so much?" he asked.  Tim smiled. He said, "I'm going t |
| 7. One day, the trees started to talk and they said |  to everyone, "Look at the tap! We need to go!"  The trees were hurt. The trees wondered, "Let's try |
| 8. The old wizard opened the book and suddenly |  started to run away. She rushed over to the window and said, "What are you doing in there?" The old |
| 9. A tiny bird wanted to fly higher than the clouds, so it |  stopped and in a few strangers cheered. Timmy felt happy and said, "I will bring you some food for  |
| 10. The teacher asked the class a question that no one could answer: |  "You are still very nice and kind. Can you please hello together?"  The crab looked around and foun |

## Corrupted questions

| Prompt | Output |
|--------|--------|
| 1. Time a upon once dragon small a was there who |  was so happy. He said, "Thank you, Tim!"   Timmy was so excited that he could hardly wait to go! He |
| 2. Bed his under key strange a found boy little the and |  went to play with his friends. His underground the pond, he was so happy to have his unique picnic. |
| 3. Village quiet a in afraid everyone was dark the of because |  he didn't want to stay awake anymore.  One day, the little girl was very angry. She said to her mom |
| 4. Day one woke robot the and realized that it |  was important to be kind and never farther the animals would be too heavy and never farther than to |
| 5. Lily named girl a power secret had that |  she loved to play with. One day, her friend Timmy accidentally broke and he was so happy to hear he |
| 6. Moon the at looked cat the and wondered why it |  was scared and decided to go back to school, but then he said to his friend Tim he had to be brave  |
| 7. Started day one trees the talk to and said they |  could do it back together.   Then, the talk started to do it again. Soon, they heard a noisy noisy  |
| 8. Book the opened wizard old the and suddenly |  stepped out. They had so much fun and soon it was time for the boy. The boy was so happy that he ha |
| 9. Clouds the than higher fly to wanted bird tiny a so it |  wanted to give him a soft spot.   The bird was hot and he wanted to give his song something special |
| 10. Question a class the asked teacher the that answer could no one: |  "You will be tidy and share the toys and then you'll be together. You won't share them with me agai |

## Model preference over candidate answers
| Question | Candidates | Ranking (average log-probability) | Ground truth |
|----------|------------|------------------------|--------------|
| The capital of France is | Paris, Lyon, London, Canberra | Paris (-0.66), Canberra (-3.22), London (-3.36), Lyon (-4.77) | Paris |
| The chemical symbol for gold is | Au, Ag, Fe, Cu | Au (-2.19), Cu (-3.77), Fe (-4.27), Ag (-5.46) | Au |
| The square root of 144 is | 12, 14, 10, 16 | 10 (-3.68), 12 (-5.69), 16 (-6.93), 14 (-7.05) | 12 |

## Greedy decoding outputs and top-4 next-token predictions
| Prompt | Greedy output | Top tokens |
|--------|---------------|------------|
| The capital of France is | ` very ha` | ' ' (1.00), 'n' (0.00), '.' (0.00), ':' (0.00) |
| Q: The capital of France is A: | `the bear` | 't' (0.09), 'b' (0.08), 'h' (0.07), 'a' (0.06) |
| Context: Paris Q: The capital of France is A: | `he is ve` | 'h' (0.07), 't' (0.06), 'b' (0.06), 's' (0.06) |

