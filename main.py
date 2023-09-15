from ai import completion
import json
import tiktoken
paragraphs = [
    """Modern politics is frequently heralded as the zenith of democratic progression. We boast of our age as one where people's voices matter more than ever, lauding the democratic values that our forebears fought for. Yet, one cannot help but notice the glaring paradox that plagues our era: while we have more mechanisms for public engagement than ever—social media, online petitions, and the like—our political landscape has never seemed more polarized.

Partisanship, rather than policy, is the fuel of modern political discourse. Parties are no longer mere vehicles for policy; they've become tribal identities. This fervent partisanship is endangering the very essence of democracy. Instead of constructive debates where the best ideas win, politics has become a zero-sum game where one's gain is another's loss.

Furthermore, the influence of money in politics muddies the waters of genuine representation. Can a politician truly represent their constituents when a hefty campaign donation is hanging over their head? Lobbying, while a legitimate part of the democratic process, often veers into the territory of undue influence, raising questions about whose interests are genuinely being represented.

Moreover, the rise of populism in various parts of the world is both a symptom of and a reaction to these flaws in the democratic system. While it speaks to genuine frustrations people have with the 'establishment', unchecked populism can endanger minority rights, leaning into majoritarian tyranny.

In conclusion, while we live in an age of unparalleled democratic tools and ideals, we must grapple with the reality of deep-seated partisanship, money-driven politics, and the double-edged sword of populism. Only by addressing these issues head-on can we hope to realize the true promise of democracy.""",
    "Mathematics, rightly viewed, possesses not only truth, but supreme beauty — a beauty cold and austere, like that of sculpture, without appeal to any part of our weaker nature, without the gorgeous trappings of painting or music, yet sublimely pure, and capable of a stern perfection such as only the greatest art can show.",    
 "Science is not only compatible with spirituality; it is a profound source of spirituality. When we recognize our place in an immensity of light-years and in the passage of ages, when we grasp the intricacy, beauty, and subtlety of life, then that soaring feeling, that sense of elation and humility combined, is surely spiritual. So are our emotions in the presence of great art or music or literature, or acts of exemplary selfless courage such as those of Mohandas Gandhi or Martin Luther King, Jr. The notion that science and spirituality are somehow mutually exclusive does a disservice to both.",    
"Mathematics is not about numbers, equations, computations, or algorithms: it is about understanding."     ,
   "Pure mathematics is, in its way, the poetry of logical ideas.",
    
"It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."    
"In my younger and more vulnerable years my father gave me some advice that I’ve been turning over in my mind ever since. 'Whenever you feel like criticizing anyone,' he told me, 'just remember that all the people in this world haven’t had the advantages that you’ve had.'"    
"“Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",    
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters.",    
"Four score and seven years ago our fathers brought forth on this continent, a new nation, "
    "conceived in Liberty, and dedicated to the proposition that all men are created equal.\n\n"
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived "
    "and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to "
    "dedicate a portion of that field, as a final resting place for those who here gave their lives that "
    "that nation might live. It is altogether fitting and proper that we should do this.\n\n"
    "But, in a larger sense, we can not dedicate — we can not consecrate — we can not hallow — this ground. "
    "The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. "
    "The world will little note, nor long remember what we say here, but it can never forget what they did here. "
    "It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here "
    "have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us — "
    "that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion — "
    "that we here highly resolve that these dead shall not have died in vain — that this nation, under God, shall have a new birth of freedom — "
    "and that government of the people, by the people, for the people, shall not perish from the earth."
]


to_chinese_system_styled = {'role': 'system', 'content': """You are a translator with a specialization in English to Chinese. Aim: Keep meaning intact so when re-translated to English, original intent is preserved."""}
to_compressed_system = {'role': 'system', 'content': """compress the user prompt text such that you (GPT) 
can reconstruct the intention of the human who wrote text with the full original intention. 
This is for yourself. It does not need to be human readable or understandable. Abuse of language mixing, 
abbreviations, symbols (unicode and emoji), or any other encodings or internal representations is all permissible, 
so long as it, used in a future prompt will yield near-identical results as the original text:"""}
to_caveman_system_styled = {'role': 'system', 'content': """You AI. Talk caveman. User give big text. You make small. Make simple."""}

systems = [
    to_chinese_system_styled,
    to_compressed_system,
    to_caveman_system_styled
]

output_data = []

print("Starting processing...")

for index, paragraph in enumerate(paragraphs):
    print(f"\nProcessing paragraph {index + 1} of {len(paragraphs)}...")
    entry = {
        "original_paragraph": paragraph
    }
    
    for system in systems:
        print(f"Using system: {system['content']}...")
        messages = [system, {'role': 'user', 'content': paragraph}]
        
        print("Sending data to OpenAI function...")
        converted_paragraph = completion(messages,model="gpt-4")
        
        print("Received converted paragraph. Storing...")
        
        if system == to_chinese_system_styled:
            entry["chinese_paragraph"] = converted_paragraph
            print("Stored Chinese converted paragraph.")
        elif system == to_compressed_system:
            entry["compressed_paragraph"] = converted_paragraph
            print("Stored Compressed converted paragraph.")
        elif system == to_caveman_system_styled:
            entry["caveman_paragraph"] = converted_paragraph
            print("Stored Caveman styled converted paragraph.")
    
    output_data.append(entry)
enc = tiktoken.encoding_for_model("gpt-4")
for index, entry in enumerate(output_data):
    print(f"\nTokenizing paragraph {index + 1} of {len(output_data)}...")
    for key, value in entry.items():
        print(f"\n{key}:")

        print(len(enc.encode(value)))

