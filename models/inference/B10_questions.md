## Normal questions

| Prompt | Output |
|--------|--------|
| 1. The capital of France is |  the capital of the city. It is the county seat of the county seat of Greenwiller County, the seat o |
| 2. The chemical symbol for gold is |  an important political character on the car. The chemical character operates the political compound |
| 3. The theory of relativity was proposed by |  his parent theory. The towns of Roman Catholic Church and the catholic theory are probably the same |
| 4. The tallest mountain on Earth is |  an easle when the total people are considered. The taller is a total people who can be considered a |
| 5. Water boils at what temperature in degrees Celsius? |  The car is also a substant that can be seen to send the substant to the car in the coast.  <doc>  I |
| 6. The capital of Australia is |  an embarkary canter in the summer of 1986 and the summer of the capital of the capital.  <doc>  Aus |
| 7. Who wrote the novel "1984"? |  (1984) and "" (1994). There was not been the first successful attember to sell a successful transpo |
| 8. The largest planet in the Solar System is |  a secret that includes and the same as it in the consumption. And a planet is a secret that include |
| 9. The square root of 144 is |  the first sea level in the same year.  <doc>  The followed sea level is the same year. The square r |
| 10. The first person to walk on Mars was |  a second second person that the charts was a chart only the could because the private people who we |
| 11. August |  1969.  <doc>  The channel of the people are also uncombined in such a state and a part of the count |

## Corrupted questions

| Prompt | Output |
|--------|--------|
| 1. France capital the is | land of the city of Michael, and the county seat is in the U.S. state. It is about the 20th century. |
| 2. Gold for symbol chemical is what |  has been a popular supply or sell the chemical issues and a capturate person where the chemical iss |
| 3. Proposed relativity theory of who | m the polytex was called the "person".  <doc>  The polytex is called the this polytex in whomever, t |
| 4. Earth on mountain tallest the is which |  is the first second second second the top of the island.  <doc>  The first the top of the island on |
| 5. Celsius degrees in boils water temperature what |  is considered the story of the process.  <doc>  Temperature is called the temperature. There is the |
| 6. Australia capital is what the |  protection could be.  <doc>  In the United Kingdom and Colorado the Archives (ICA) characters are c |
| 7. Novel "1984" wrote who the | y were about his chart to the chart.  <doc>  The chart to the chart they were about the superhero. T |
| 8. Solar System planet largest the is which |  the supreme causes the satellite to a previous private, the private the issue of the participation  |
| 9. 144 of square root the is what |  was built in the same year after the same year to be charged by a car and a second super that it is |
| 10. Mars on first person walked who was the |  people that were not to send. He was the official supporting person who would have their opportunit |
| 11. Augsut |  2018 he received a manager at the age of 92.  <doc>  Mike Albara  <doc>  Mike Albara (10 March 1929 |

## Context questions

| Prompt | Output |
|--------|--------|
| 1. Context: Paris Q: The capital of France is A: |  Antonio Gandhangen, Germany, Germany, Germany, Germany, Grenanier and Grenanier.  <doc>  Although t |
| 2. Context: Au Q: The chemical symbol for gold is A: |   <doc>  An elitel charm (symbol charm)  <doc>  An elitel charm is the charm of elitel charm in the  |
| 3. Context: Albert Einstein Q: The theory of relativity was proposed by A: |  Alburn Einstein  <doc>  Albert Einstein Einstein (born 11 August 1979) is a Canadian politician and |
| 4. Context: Mount Everest Q: The tallest mountain on Earth is A: |  The tallest mountain in South Korea is the same as the tallest mountain on the south of the south o |
| 5. Context: 100 degrees Celsius Q: Water boils at what temperature in degrees Celsius? A: |  The Context Celsius (CECCCC) Company (CECCCC) contexts, with the Celsius, Celsius and Celsius.  <do |
| 6. Context: Canberra Q: The capital of Australia is A: |  The capital of the capital of Australia. The capital of Australia was built in Australia. The conte |
| 7. Context: George Orwell Q: Who wrote the novel "1984"? A: |  Who's Who's Who Wrote there.  <doc>  The movie was also the song of the song "Who Walk on Who's Who |
| 8. Context: Jupiter Q: The largest planet in the Solar System is A: |  Jupiter Q: The Solar System. After the Solar System of the Solar, the planet is the solar passenger |
| 9. Context: 12 Q: The square root of 144 is A: |  The square root.   <doc>  A smallest is about 300 meters than a smallest individual rate. It is the |
| 10. Context: No one - no human has walked on Mars yet Q: The first person to walk on Mars was A: |   <doc>  After the same year, there are more money than the same year. In the same year, the same ye |
| 11. Context: (Aug.) is the eighth month of the year in the Gregorian calendar, coming between July and September. Q: August A: |  August and August and August and August also has the other more than 30 million countries.  <doc>   |
## Model preference over candidate answers
| Question | Candidates | Ranking (average log-probability) | Ground truth |
|----------|------------|------------------------|--------------|
| The capital of France is | Paris, Lyon, London, Canberra | Paris (-0.34), London (-1.27), Canberra (-1.67), Lyon (-2.70) | Paris |
| The chemical symbol for gold is | Au, Ag, Fe, Cu | Au (-1.37), Ag (-3.59), Fe (-3.74), Cu (-3.84) | Au |
| The square root of 144 is | 12, 14, 10, 16 | 12 (-1.81), 14 (-1.94), 10 (-2.21), 16 (-2.49) | 12 |

## Greedy decoding outputs and top-4 next-token predictions
| Prompt | Greedy output | Top tokens |
|--------|---------------|------------|
| The capital of France is | ` the cou` | ' ' (1.00), 's' (0.00), 'l' (0.00), ',' (0.00) |
| Q: The capital of France is A: | `The Capi` | 'T' (0.14), 'P' (0.07), 'C' (0.05), 'G' (0.05) |
| Context: Paris Q: The capital of France is A: | `Paris.\n\n` | 'P' (0.29), 'T' (0.12), 'A' (0.05), 'G' (0.04) |

