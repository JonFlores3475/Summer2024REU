import discord
from discord.ext import commands
import re
import random
# import aiohttp

CHANNEL_ID = 1228067170282504284
BOT_TOKEN = 'MTIyODA1NzUyNTQyNTgwMzI3NA.GWaL1B.0HM4JrAH6W4MUo5H35q0ik-OfqOg_SFZjk4V24'

intents = discord.Intents.default()
intents.members = True  # NOQA

processed_messages = set()

bot = commands.Bot(command_prefix='', intents=intents.all())

@bot.event
async def on_ready():
    print('AndreBot is on the Job!')
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

@bot.command(aliases=['hi', 'hey', "what's up", 'hola', 'Hello', 'HI!', 'HEY', 'HEY!', 'hello'])
async def helo(ctx):
    await ctx.send("Hello! Welcome to my Classroom. How can I help you?")

# guide command ------------------------------------------------------------------------------------------------
@bot.command()
async def guide(ctx):
    await ctx.send("Here is what I can do:\n"
                   "1. I can talk to you about programming languages\n"
                   "2. I can play rock, paper, scissors - rps [tool]\n"
                   "3. I can do some basic math - add [x y, ..] sub [x y z z], mul [x y z], div [x y z]\n"
                   "4. I can tell you some of my finest dad jokes - joke\n"
                   "5. I can play magic 8 ball with you - 8ball [question]?\n"
                   "6. I can flip a coin for you - coin\n"
                   "7. I can play trivia with you - trivia\n"
                   "8. I can show you my selfie - peekaboo\n")

# trivia command ------------------------------------------------------------------------------------------------
@bot.command()
async def trivia(ctx):
    questions = [
        "What is the best coding language?",
        "What language is most interpreters built upon logic wise?",
        "In object-oriented programming, what is the process of creating an instance of a class called?",
        "What programming language that we use in class uses automatic memory management through garbage collection?",
        "Which programming language is known for its simplicity, concurrency support, and built-in networking?",
        "What language is often used for scripting, web development, and server-side development, known for its asynchronous and event-driven architecture?",
        "What programming language is often used for statistical computing, data analysis, and machine learning?",
        "Which language is famous for its use in game development and has a syntax similar to C?",
        "What is the primary programming language used for developing Android applications?",
        "Which language is often used for system administration, network programming, and automation?",
    ]
    answers = [
        "Go",
        "Lisp",
        "Instantiation",
        "Java",
        "Python",
        "JavaScript",
        "R",
        "C++",
        "Java",
        "Python",
    ]
    index = random.randint(0, len(questions) - 1)
    question = questions[index]
    answer = answers[index]
    await ctx.send(f"Trivia Question: {question}")
    try:
        guess = await bot.wait_for("message", timeout=15, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
        if guess.content.lower() == answer.lower():
            await ctx.send("Correct! 🎉")
        else:
            await ctx.send(f"Sorry, the correct answer was: {answer}")
    except asyncio.TimeoutError:
        await ctx.send(f"Time's up! The correct answer was: {answer}")

## rps (rock , paper, or scissors) -----------------------------------------------------------------------------------------------
@bot.command()
async def rps(ctx, tool):
    rpsGame = ['rock', 'paper', 'scissors']

    comp_choice = random.choice(rpsGame)
    if tool == 'rock':
        if comp_choice == 'rock':
            await ctx.send(f'Well, that was weird. We tied.\nYour choice: {tool}\nMy choice: {comp_choice}')
        elif comp_choice == 'paper':
            await ctx.send(f'Nice try, but I won that time!!\nYour choice: {tool}\nMy choice: {comp_choice}')
        elif comp_choice == 'scissors':
            await ctx.send(
                f"Aw, you beat me. It won't happen again!\nYour choice: {tool}\nMy choice: {comp_choice}")

    elif tool == 'paper':
        if comp_choice == 'rock':
            await ctx.send(
                f'The pen beats the sword? More like the paper beats the rock!!\nYour choice: {tool}\nMy choice: {comp_choice}')
        elif comp_choice == 'paper':
            await ctx.send(
                f'Oh, wacky. We just tied. I call a rematch!!\nYour choice: {tool}\nMy choice: {comp_choice}')
        elif comp_choice == 'scissors':
            await ctx.send(
                f"Aw man, you actually managed to beat me.\nYour choice: {tool}\nMy choice: {comp_choice}")

    elif tool == 'scissors':
        if comp_choice == 'rock':
            await ctx.send(
                f'HAHA!! I JUST CRUSHED YOU!! I rock!!\nYour choice: {tool}\nMy choice: {comp_choice}')
        elif comp_choice == 'paper':
            await ctx.send(f'Bruh. >: |\nYour choice: {tool}\nMy choice: {comp_choice}')
        elif comp_choice == 'scissors':
            await ctx.send(f"Oh well, we tied.\nYour choice: {tool}\nMy choice: {comp_choice}")

# math commands ------------------------------------------------------------------------------------------------
# adds any number of values together
@bot.command()
async def add(ctx, *arr):
    val = 0
    for i in arr:
        val += int(i)

    await ctx.send(val)

# subtracts any number of values together
@bot.command()
async def sub(ctx, *arr):
    if not arr:
        await ctx.send("No numbers provided.")
        return

    val = int(arr[0])  # Initialize val with the first number
    for num in arr[1:]:
        val -= int(num)  # Subtract each subsequent number from val

    await ctx.send(val)

# multiplies any number of values together
@bot.command()
async def mult(ctx, *arr):
    if not arr:
        await ctx.send("No numbers provided.")
        return

    val = 1  # Initialize val with 1
    for num in arr:
        val *= int(num)  # Multiply each number with val

    await ctx.send(val)

# divides any number of values together
@bot.command()
async def div(ctx, *arr):
    if not arr:
        await ctx.send("No numbers provided.")
        return

    val = int(arr[0])  # Initialize val with the first number
    for num in arr[1:]:
        if int(num) == 0:
            await ctx.send("Cannot divide by zero.")
            return
        val /= int(num)  # Divide val by each subsequent number

    await ctx.send(val)

@bot.command(aliases=["flip a coin", "heads or tails", "flip"])
async def coin(ctx):
    flip = ['heads', 'tails']
    await ctx.send(random.choice(flip))

# 8ball command ------------------------------------------------------------------------------------------------
@bot.command(aliases=['8ball', 'eightball', '8-ball', 'eight-ball'])
async def eight_ball(ctx, *, question):
    responses = [
        "It is certain.",
        "It is decidedly so.",
        "Without a doubt.",
        "Yes - definitely.",
        "You may rely on it.",
        "As I see it, yes.",
        "Most likely.",
        "Outlook good.",
        "Yes.",
        "Signs point to yes.",
        "Reply hazy, try again.",
        "Ask again later.",
        "Better not tell you now.",
        "Cannot predict now.",
        "Concentrate and ask again.",
        "Don't count on it.",
        "My reply is no.",
        "My sources say no.",
        "Outlook not so good.",
        "Very doubtful."
    ]

    response = random.choice(responses)
    await ctx.send(f"Question: {question}\nAnswer: {response}")

# Message responses --------------------------------------------------------------------------------------------
@bot.event
async def on_message(message):
    # Check if the message has already been processed
    if message.id in processed_messages:
        return
    processed_messages.add(message.id)

    # Joke responses - - - - - - - - - - - - - - - - - - - - - 
    if ("joke" in message.content.lower() or "funny" in message.content.lower() or "laugh" in message.content.lower() or "silly" in message.content.lower() or "dad" in message.content.lower()) and message.author != bot.user:
        jokes = ["Did you hear about the cat that ate a lemon? Now it’s a sour puss.",
                 "How do mice floss their teeth? With string cheese.",
                 "What do you call a happy cowboy? A jolly rancher" ,
                 "What did one wall say to the other? I’ll meet you at the corner.",
                 "What do clouds wear beneath their pants? Thunderwear.",
                 "What kind of bagel can travel? A plain bagel.",
                 "When's the best time to call your dentist? Tooth-hurty.",
                 "What's the best way to catch a fish? Ask someone to throw it to you.",
                 "What do you call a cat with eight legs? An octo-puss.",
                 "What do you call an anxious fly? A jitterbug.",
                 "What did the potato chip say to the other? Let's go for a dip.",
                 "Why shouldn't you tell jokes to a duck? Because they'll quack up.",
                 "How did the piano get locked out of its car? It lost its keys.",
                 "Why did the orchestra get struck by lightning? It had a conductor.",
                 "What do you call a fake dad? A faux pas.",
                 "How do you make an eggroll? You push it.",
                 "I've never been a fan of facial hair. But now it's starting to grow on me.",
                 "Did you hear about the fire at the shoe factory? Unfortunately, many soles were lost.",
                 "What do you call a pig who knows how to use a butcher knife? A pork chop.",
                 "What kind of fish knows how to do an appendectomy? A Sturgeon.",
                 "How do you hire a horse? Put up a ladder.",
                 "Why did the pony ask for a glass of water? Because it was a little horse.",
                 "Is there anything worse than when it's raining cats and dogs? Yes, hailing taxis."
                 ]
        await message.channel.send(random.choice(jokes))

    # Filter out cuss words
    cuss_words = ['fuck', 'shit', 'bitch', 'heck', 'darn', 'damn', 'ass', 'crap', 'hell', 'frick', 'fricking',
                  'fucking']
    mean_words = ['loser', 'stupid', 'bum', 'jerk', 'meanie', 'dumb', 'fool', 'fricker']
    sleep_words = ['tired', 'sleep', 'sleeping', 'awake', 'snooze', 'bed', 'bedtime', 'sleepy']
    if any(re.search(r'\b' + re.escape(word) + r'\b', message.content.lower()) for word in cuss_words):
        await message.channel.send("Haha, no cussing in my discord class.")
        return
    # mean words
    if any(re.search(r'\b' + re.escape(word) + r'\b', message.content.lower()) for word in mean_words):
        await message.channel.send("That's not a nice word... Let's make better choices.")
        return
    # tired
    if any(re.search(r'\b' + re.escape(word) + r'\b', message.content.lower()) for word in sleep_words) and message.author != bot.user:
        await message.channel.send(
            "You might have heard about it now, but I have two little kids, so I obviously don't get much sleep either. ")
        return

    # Scheme responses - - - - - - - - - - -- - - - - - - - - - - - - - 
    if "scheme" in message.content.lower() and "hard" in message.content.lower():
        response = (
            "One day when you are coding you’ll realize you think of a recursive program before an iterative solution… "
            "that’s when you’ll think back on the scheme project and realize it was all worth it.")
        await message.channel.send(response)
    elif "language is scheme" in message.content.lower():
        response = "Scheme is a strongly typed language, but it’s dynamically checked."
        await message.channel.send(response)
    elif ("functions" in message.content.lower() and "scheme" in message.content.lower()) or (
            "variable" in message.content.lower() and "scheme" in message.content.lower()):
        response = (
            "You may be sitting in your seat right now while I make a comment like, well, functions are just variables, "
            "but you can’t take that for granted!")
        await message.channel.send(response)
    elif "different" in message.content.lower() and "scheme" in message.content.lower():
        response = "Scheme is not at all like you’ve been taught to think about programming."
        await message.channel.send(response)
    elif ("hate" in message.content.lower() and "scheme" in message.content.lower()) or (
            "suck" in message.content.lower() and "scheme" in message.content.lower()):
        response = "I am very passionate about you having this experience, even if you are kicking and screaming the whole way through."
        await message.channel.send(response)
    elif "scheme" in message.content.lower() and "fun" in message.content.lower():
        response = "If you want to hear more about Scheme, I really recommend Structure and Interpretation of Computer Programs! It’s also known as the wizard book :)"
        await message.channel.send(response)

    # Response to questions
    elif "?" in message.content or "help" in message.content.lower():
        if "scheme" in message.content.lower():
            scheme_responses = [
                "Have you tried writing your problem out on a whiteboard yet?",
                "Make sure to read the project specifications!",
                "It’s all just dealing with lists!",
                "Do you have enough parenthesis?"
            ]
            response = random.choice(scheme_responses)
            await message.channel.send(response)

    # Prolog responses - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if "prolog" in message.content.lower():
        # Response to specific statements
        if "explain" in message.content.lower():
            response = "Prolog is … a great language that displays and uses relations between objects to accomplish a task."
            await message.channel.send(response)
        elif "confusing" in message.content.lower():
            response = "If you don’t give Prolog a clear path, it will use backtracking and find its own."
            await message.channel.send(response)

        # Response to questions
        elif "?" in message.content or "help" in message.content.lower():
            prolog_responses = [
                "Uhhhh, I’m not quite sure… let’s take a look at it more closely.",
                "Lets work through it together!",
                "Well, did you put a period at the end of it?",
                "Make sure to read the project specifications!"
            ]
            response = random.choice(prolog_responses)
            await message.channel.send(response)

    # Go responses - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if "go" in message.content.lower() and message.author != bot.user:
        # Response to specific statements
        if "like" in message.content.lower() or "love" in message.content.lower() or "enjoy" in message.content.lower() or "cool" in message.content.lower():
            response = "I like Go a lot, too!"
            await message.channel.send(response)
        elif "functionality" in message.content.lower():
            response = "Using Go is indeed like having the functionality of Java threads without the headache."
            await message.channel.send(response)
        elif "garb" in message.content.lower():
            response = "Yes, Go has garbage collection, and it's quite helpful!"
            await message.channel.send(response)
    
        # Response to questions
        elif "?" in message.content.lower() or "help" in message.content.lower():
            go_responses = [
                "When you are looking at Go, think about it like C++ functionality with Python syntax.",
                "Make sure to read the project specifications!"
            ]
            response = random.choice(go_responses)
            await message.channel.send(response)

    # JavaScript responses - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if "javascript" in message.content.lower() and message.author != bot.user:
        # Response to specific statements
        if "goofy" in message.content.lower() or "weird" in message.content.lower() or "confusing" in message.content.lower():
            response = "It's true, JavaScript can be a bit goofy sometimes!"
            await message.channel.send(response)
        elif "like" in message.content.lower():
            response = "I actually really like JavaScript, too, despite its quirks."
            await message.channel.send(response)
        elif "application" in message.content.lower():
            response = "Ah JavaScript, the language of endless possibilities and headaches."
            await message.channel.send(response)
        
        # Response to questions
        elif "?" in message.content.lower() or "help" in message.content.lower():
            js_responses = [
                "Remember, in JavaScript, '==' is not always equal to '==='.",
                "When in doubt, console.log() it out!",
                "Make sure to read the project specifications!"
            ]
            response = random.choice(js_responses)
            await message.channel.send(response)

    # Interpreter Assignment (IA) responses - - - - - - - - - - - - - - - - - -  -- - - - ----------
    if (("interpreter" in message.content.lower() or "ia" in message.content.lower()) and message.author != bot.user):
        # Response to specific statements
        if "function" in message.content.lower():
            response = "Well, the interpreter is basically us making Scheme!"
            await message.channel.send(response)
        elif "use" in message.content.lower():
            response = "You have now made your own coding program… isn’t that cool, guys!"
            await message.channel.send(response)
        elif "hard" in message.content.lower():
            response = "And now we have just made a really bad and inefficient version of Lisp."
            await message.channel.send(response)
        
        # Response to questions
        elif "?" in message.content.lower() or "help" in message.content.lower():
            ia_responses = [
                "Everyone who attempts to make their own interpreter without understanding Lisp is doomed to make a really bad and inefficient version of Lisp.",
                "Make sure to test everything in your environment.",
                "Have you tried working through the parse tree?"
            ]
            response = random.choice(ia_responses)
            await message.channel.send(response)

    #Random Responses  - - -- - - - ---- - - - - - -  - - --  --- - -----  --- - 
    # Regular Expressions
    if ("regular expr" in message.content.lower() or "regex" in message.content.lower()) and message.author != bot.user:
        response = ("If you get excited about regular expressions, first of all, I'm flattered, "
                    "second of all, they are nightmarish things.")
        await message.channel.send(response)

    # Merge Conflict
    elif "merge" in message.content.lower() and message.author != bot.user:
        response = "Ah the time capsule of code, with great power comes… the occasional merge conflict."
        await message.channel.send(response)

    # Spring Break
    elif "spring break" in message.content.lower() and message.author != bot.user:
        response = "I’ll plan everything to be mostly done before spring break, unless I don’t."
        await message.channel.send(response)

    # Class
    elif "classroom" in message.content.lower() and message.author != bot.user:
        response = "You better believe I have a thermometer for you today!"
        await message.channel.send(response)

    # Final Project
    elif "final project" in message.content.lower() and message.author != bot.users:
        final_project_responses = [
            "Are you so excited to present in class?",
            "It doesn’t have to be good, just please do not come with nothing."
        ]
        response = random.choice(final_project_responses)
        await message.channel.send(response)

    if "favorite student" in message.content.lower() and message.author != bot.user:
        response = "I can't pick favorites, I love all of my students equally. But if I had to pick, I would have to pick Zeke."
        await message.channel.send(response)

    # andre's experience - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if "intern" in message.content.lower() and message.author != bot.user:
        response = ("I was an Intern at HP for about a year from May 2014 to June 2015. That internship helped me "
                    "land a job at HP for almost 8 years")
        await message.channel.send(response)
    elif ("work" in message.content.lower() or "job" in message.content.lower()) and message.author != bot.user:
        response = ("At HP, I started out as a an Engineering and Quality Assurance Lead. I was in that position "
                    "for 4 years until I got promoted. My next role at HP was a level 1 project manager. It was "
                    "not until November of 2020 until I moved up to a level 2 project manager. I left HP to "
                    "become a professor as Boise State. I enjoy helping young minds prepare for their future in "
                    "the tech industry!")
        await message.channel.send(response)
    elif "live" in message.content.lower() and message.author != bot.user:
        response = ("I live here in Boise, right by campus. I am excited to ride my bike to"
                    " work when the weather is nice")
        await message.channel.send(response)

    await bot.process_commands(message)

# Selfie command ------------------------------------------------------------------------------------------------
@bot.command(hidden=True)
async def peekaboo(ctx):
    """Here is my selfie."""
    art = """                       
    *&&&    ,#(//////(/***,,,,,,,*****///(((((((#(((/(///(((#(((((#%,   *&%/    
   ,&&&.    ,(((###((**,,,,,,,,,,,,,,,,***************,**///((((((##     *&&/   
   %&@*     .##%&%##/*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*,,,***/((##(#*     (&&.  
  ,&&&.     /%%%%%%#/,,,,,,,,,....,,,,,,,,,,,,,,,,.......,,,,*(#####(    ,&&&(  
  ,&&&%    .(%%%%%#(/,,,,,,,,......,,,,,,,,,,,,,..........,,,*/####%%@&#&@@@&   
   .&&&%,,(&%&&%%%#/,,,,,,*****,,,,,,,,,,,,,,,,,,,*******,,,,,*(%#%&&&%%@&&%.   
     (#%&&%@&&&&&%(*,,*/(###%%%%%#((//********/(#####%%%%&&&&%((%&%&&@&&&#%%,   
     *%%%&(%&&&&&###%&#(((((###%%#####&@&&&@@&#((######(//(((#&&@&@&&@&@&%#*.   
      #&&&@&@&&&%%%@@***/(#%(#%&%((##((&@%(%@#((((#%&%((%#((//#&&#&&&@&@%%#.,.  
      .%&&&@@@&&&&(*@*,**((/(((((((((((@/...*&///((////////*,*%&#@&&@%&&%%* *   
     ..,&&&@&@@@&@#*/#,,,**////(/(///*##,,.../&***//////***,,/%(#@&&@%@&&% .*   
     .. /&&&&&@@@@@**(/,,,,,,,,,,*,,,##,,,...,*&#**,,,**,,,,/%(*&@@@@@@@@/./.   
      ...%@&&@@@@@@%,,,*(#%%%%%%%#(///*,......,*/*****,,......*(@@@@@@@@&*/*    
       ,,,@@@&@@@@@@/,,,,.,,,,***//(,,,,,,,,...,.*//*,,......,,&@@@@@@@@#//     
        ,*/@@@@@@@@@%*,,,,,***/////(((#(//*///((/**/(/**,,,,,,*@@@@@@@@&#*      
         ,(#@@@@@@@@@/*******/////////(((((((//*****/((/***,**#@@@@&@&&%,       
           *#%%##@@@@&******///(/////*////*****,,,,**//((****/@@%/..&*          
                    ,/*******/(/(###(#(#((((*///**#&%(///***,      .*           
                       ,*******///(#((//,,....**/(/****/****       /            
                    .%@&*********////((/(((((//************,      ./               
"""
    await ctx.send("```" + art + "```")  

# member join/leave event listeners ----------------------------------------------------------------------------
@bot.event
async def on_member_join(member):
    """
    Event listener for when a member joins the server.
    Sends a welcome message to the designated text channel within the specified category.
    """
    category = bot.get_channel(CHANNEL_ID)  # Get the category channel
    if category and isinstance(category, discord.CategoryChannel):
        # Get the first text channel within the category (you can adjust this as needed)
        text_channel = category.text_channels[0] if category.text_channels else None
        if text_channel:
            # Send a welcome message to the text channel
            await text_channel.send(f"Hello! Welcome to my Classroom {member.mention}!")
        else:
            print("No text channels found in the category.")
    else:
        print("Category channel not found.")

@bot.event
async def on_member_remove(member):
    """
    Event listener for when a member leaves the server.
    Sends a farewell message to the designated text channel within the specified category.
    """
    category = bot.get_channel(CHANNEL_ID)  # Get the category channel
    if category and isinstance(category, discord.CategoryChannel):
        # Get the first text channel within the category (you can adjust this as needed)
        text_channel = category.text_channels[0] if category.text_channels else None
        if text_channel:
            # Send a farewell message to the text channel
            await text_channel.send(f"Sorry to see you go {member.mention}. Have a good rest of your day. Goodbye!")
        else:
            print("No text channels found in the category.")
    else:
        print("Category channel not found.")

bot.run(BOT_TOKEN)