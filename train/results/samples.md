# Samples

Unedited output of `trit generate` with `models/tinystories-15m.trit` (v0.1.0). Each block is the exact command and everything it printed to stdout; nothing was picked or retried.

## Default settings (temperature 0.8, top-p 0.9), seeds 1 to 4

```
$ trit generate --seed 1
Once upon a time, there was a little girl named Sue. Sue loved to play with her toys. One day, she found a big box. Inside the box, there was a surprise. Sue was very excited.
Sue wanted to make a toy car with her new gear. She put the gear in a bag and went outside. Sue played with her gear all day long. She made the gear go fast and slow. It went round and round. Sue was so happy with her new toy.
But then, something unexpected happened. Sue's gear went far away. It went into a deep hole. Sue was sad and scared. She did not know what to do. Then, she saw a big, friendly dog. The dog had the gear in its mouth! The dog gave the gear back to Sue. Sue was so happy and played with her new dog friend all day.
```

```
$ trit generate --seed 2
Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. One day, he saw a shiny scale on the floor. Tim was very happy.
Tim wanted to show his friend, Sam. He took the scale and went to Sam's house. Sam was a bit grumpy. They played with the scale, but they did not know that the scale was magic.
Tim and Sam played with the glowing car all day. They took turns using the magic scale. They laughed and had lots of fun. When it was time to go home, Tim put the magic scale back on the shelf. They were not grumpy anymore.
```

```
$ trit generate --seed 3
Once upon a time, there was a little girl named Mia. Mia had a toy bear named Mr. Bear. They were best friends. One day, Mia found a fake rock in the park. She picked it up and showed it to Mr. Bear.
"Look, Mr. Bear! I found a fake rock!" Mia said. Mr. Bear looked at the fake rock and said, "Wow! What a nice rock! Let's play with it!" They took turns throwing the fake rock into the air and laughing.
As they played, a big wind came and blew the fake rock away. Mia and Mr. Bear were sad. They looked for the fake rock, but they could not find it. Then, Mia had an idea. She would make a new game.
Mia and Mr. Bear looked and looked. They found a big, soft pillow. They put the fake rock on the pillow and covered it with soft pillows. The wind could not blow the fake rock away. Mia and Mr. Bear were happy again. They played with the pillow all day long.
```

```
$ trit generate --seed 4
Once upon a time, in a big forest, there was a little girl named Lily. Lily had a big smile and loved to play with her friends. One day, while playing, she saw a big mushroom. She thought it was very pretty and wanted to touch it.
Lily asked her friend, a small bird, "Can I touch the mushroom?" The bird said, "No, it is too big. You can't touch it." Lily felt sad but she did not listen to the bird. She touched the mushroom and it started to grow bigger. It became so big that it made a lot of smoke.
The smoke made the forest very sad. Lily and the bird did not know what to do. They were stuck in the smoke. They tried to help, but it was too late. The forest was still full of smoke, and the animals were stuck in the forest. In the end, Lily and the bird were very sad.
```

## Other prompts

```
$ trit generate --prompt 'The dragon was sad because' --seed 1
The dragon was sad because he was different. He was really lonely and wished he had a friend. He decided to stay and play with the other dragons. 
The dragon was so happy to have a friend that he had the same. He spent every day playing games, but he was also very lonely. 
Then the dragon had a great idea. He decided to go on an adventure. He packed his bag and set off. 
When he came back, he was surprised to see the dragon. The dragon was so happy to see him! 
The dragon was so happy that he was no longer sad. He was now his own friend.
```

```
$ trit generate --prompt 'Lily found a tiny door in the garden.' --seed 1
Lily found a tiny door in the garden. She was so happy! She put the door in her pocket and ran to show her mom.
"Mom, look what I found!" Lily said, pointing to the door.
Her mom smiled and said, "Wow, that's so tiny! What a cute door you found!"
Lily was so excited. She wanted to see what was inside the tiny door. She opened it and found a room full of toys!
Lily was so happy. She had found a place she could play with all her new toys. She played and laughed until it was time to go home.
```

```
$ trit generate --prompt 'One day, a little robot' --seed 1
One day, a little robot named Robby went for a walk. Robby was a big, happy robot. He liked to play with his friends. They would run, jump, and laugh together.
One day, Robby met a new friend. His name was Tom. Tom was very reliable. He always helped Robby when he needed to. Robby and Tom liked to play together. They were best friends.
One day, Robby and Tom were playing a game. They wanted to see who could make the most sounds. Robby tried to make a sound, but Tom was too strong. They laughed and played for a long time. Robby and Tom were very happy.
```

## Greedy decoding (temperature 0)

```
$ trit generate --temperature 0 --seed 1
Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. One day, he saw a big, red car in the store. He wanted to buy it, but he did not have enough money.
Tim asked his mom, "Can I have the big, red car, please?" His mom said, "Yes, you can have it, but you must be careful with it." Tim was very happy and took the car home.
At home, Tim played with his new car all day. He made it go fast and slow, and he was very careful. He made the car go fast and slow, and he was very careful with it. Tim and his mom had lots of fun playing with the big, red car.
```

## Too hot (temperature 1.4, top-p 1)

```
$ trit generate --temperature 1.4 --top-p 1 --seed 1
Once upon a time, there lived a sweet young prince and a clever beenakowabried a At wave orondQuzzle.
The prince and reward were badly surprised to trad carsash beh jealously. CMollycolored faces comfortable in the peoplelight they had been out giving empty parents. HerJohn cut his hheves to make arm view them so special and heal only the same rout balloon race compassionless.
Instead, the father moral to respect ben& would start over about more parJoe explore, packages in the horizon, often shared them with another person insectly.
Moral of the story: Sauding a happy kid than selfish looks afterinesst flag your incredible Friend was that splashing liked exciting things. Don-lophones canally bloom se watr much more, incredible things.
```

## Off the training distribution

```
$ trit generate --prompt 'The stock market fell sharply today as investors' --seed 1
The stock market fell sharply today as investors. Everybody was in pain and sad. One day, a little girl saw a bright yellow bird in the market. She asked the bird why it was so pretty. The bird said, "I was just trying to take some of your grabbing my dress, but I'm still busy with my eggs." 
The little girl smiled and said, "Don't worry. I will help you. We'll make a trade." The girl and the bird agreed and worked together to make the trade. 
At the end of the trade, the girl was sad. But the bird said, "It's ok. We can still make a trade. Let's make a big, bright yellow box together!" 
The girl smiled and said, "Yes, let's do it!" They worked together and soon the girl had made a beautiful, yellow box. 
The bird was so happy and said, "I'm glad you had such a good help!" 
The girl smiled and said, "Yes! We did a great job!"
```
