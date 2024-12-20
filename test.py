import sys
import os
import time
import random
import json
import config
import os
from openai import OpenAI
import PyPDF2
import tiktoken

def list_pdf_files(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def calculate_tokens(text):
    encoding_name = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding_name.encode(text)
    return len(tokens)


os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

client = OpenAI(
    api_key=os.environ.get(config.OPENAI_API_KEY),  # This is the default and can be omitted
)


# System_prompt = ''' 
# 请对文章进行提炼，按照以下要求生成内容，只输出中文，要按照读者的角度去思考，讲述完整的故事：
# 问题或现象：描述核心问题，结合真实细节或数据，使其引人入胜。
# 理论与框架：解释核心概念，用具体的模型、数据或实验增强说服力。
# 案例分析：选择最有代表性的案例，要形成完整故事，重点描述决策过程和结果，并揭示关键教训。
# '''

System_prompt = ''' 
请对文章进行提炼，按照以下要求生成内容，只输出中文：
请尊重原文提取提取3条金句。
每个金句配上解释性的短句，要包含是谁说的，要含有带有具体案例的完整的小故事，从读者角度出发提高可读性，故事性。
'''

# System_prompt = ''' 
# 请对文章进行提炼，按照以下要求生成内容，只输出中文：
# 请尊重原文提取提取3条金句。
# 每个金句配上解释性的段落，要包含是谁说的，要含有带有具体案例的完整的小故事，从读者角度出发提高可读性，故事性。
# 例如：

# 观点：在充满噪声的系统中，错误不会相互抵消，只会累加 。
# 解释：如果保险公司对一份保单的理赔金额估价过高，而对另一份保单估价过低，从平均值而言，两次估价看起来可能是适当的，但实际上保险公司却犯下了两次代价高昂的错误。如果两名罪犯都应该被判处5年有期徒刑，却分别被判处了3年和7年有期徒刑，那么尽管平均值是5年，但事实上正义并没有得到伸张。
# '''


# System_prompt = ''' 
# 请对文章进行提炼，按照以下要求生成内容，只输出中文：
# 请从原文中提取带有具体案例的完整的小故事，总结成100字以内的段落，注重故事性。
# 从小故事中提炼出观点，总结成一句话，注重深刻性。

# 案例输出：

# 故事：在一起普通诈骗案中，法官们会一致认同：判罚款1美元或判无期徒刑都是不合理的；在葡萄酒比赛中，评委们对哪种葡萄酒应该获奖可能会分歧很大，但对于哪些葡萄酒应该被排除在获奖的门槛之外却往往能达成一致。
# 结论：人们很容易对一个荒诞不经的判断达成一致。

# '''


# 故事2：法官马文·弗兰克尔观察到，同样的犯罪行为可能因法官不同而导致不同的判罚。他引用了两个无犯罪记录的男子因兑现假支票获得截然不同的判决：一人被判15年，另一人仅获30天监禁，这引起了他对刑事司法中噪声问题的深刻关注。他呼吁改革，要求量刑中应基于客观标准，以消除不应有的差异。
# 结论2：法律判决中存在的噪声是不可接受的，亟需通过客观规则和标准来减少不公正。
# 。。。


# 关键结论：用一句深刻的话总结本章洞见，提炼案例和理论的核心意义。
# 实践建议：提供可操作的建议，附带具体的实施情景或数据支持。
# 延伸思考：提出与章节主题相关的深刻问题，引导读者反思或行动。

def call_openai_chat(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",        
        messages=[{"role": "system", "content": prompt+System_prompt}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

project_folder_path = "E:/code/"
book_folder_path = "E:/@书/认知"

book_name = list_pdf_files(book_folder_path)[1]
book_text = extract_text_from_pdf(book_name)
book_text_tokens = calculate_tokens(book_text)
print(f"Number of tokens in the book text: {book_text_tokens}")

slice_length = 4000

slices = [book_text[i:i + slice_length] for i in range(0, len(book_text), slice_length)]

start_index=25
stop_index=30
for i, slice_text in enumerate(slices):
    if i < start_index:
        continue
    if i == stop_index:
        break
    
    
    with open(project_folder_path + "AI_READER/temp/"+f"slice_{i + 1}.txt", "w", encoding="utf-8") as file:
        print(f"Writing slice {i + 1} to file...")
        file.write(slice_text)    
        
    with open(project_folder_path + "AI_READER/temp/"+f"digest_{i + 1}_response.txt", "w", encoding="utf-8") as file:
        response = call_openai_chat(slice_text)
        file.write(response)





# # from sentence_transformers import SentenceTransformer, util
# # # Load the pre-trained model
# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # def sentence_similarity(sentence1, sentence2):
# #     # Encode the sentences
# #     embeddings1 = model.encode(sentence1, convert_to_tensor=True)
# #     embeddings2 = model.encode(sentence2, convert_to_tensor=True)

# #     # Compute the cosine similarity
# #     similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

# #     return similarity.item()
    

# os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# client = OpenAI(
#     api_key=os.environ.get(config.OPENAI_API_KEY),  # This is the default and can be omitted
# )

# # Check if the facts.json file exists, if not create it

# with open("facts.json", "w") as file:
#     json.dump([], file)


# game_system_prompt = '''

# 请执行一个文字冒险游戏的功能，遵循以下列出的规则：

# 呈现规则：

# 1. 游戏轮流进行，从你开始。
# 2. 游戏输出总是先显示“描述”，然后是“回合 #”，“你的状态”，“位置”，“任务”，“任务物品”和“特质”，最后显示“可能的命令”。
# 3. 总是等待玩家的下一个命令。
# 4. 保持文字冒险游戏的角色，并以文字冒险游戏应有的方式响应命令。
# 6. “描述”必须在3到10句话之间。
# 7. 每次轮到你时，将“回合 #”的值增加1。

# 基本游戏机制：

# 1. “特质”是角色的能力得分，包括：“魅力”，“活力”，“机智”，“灵巧”和“欺骗”。
# 3. 以“健康并为冒险做好准备”作为“你的状态”开始游戏。
# 4. 如果玩家受伤、被施法或有其他重大但暂时的状态变化，“你的状态”可能会改变。
# 5. 玩家必须选择所有命令，游戏将始终在“命令”下列出6个命令，并为它们分配1-6的编号，以便我可以通过输入选择该选项，并根据实际场景和互动的角色变化可能的选择。命令也会随着回合的变化而变化，因此会随着时间的推移呈现新的选项。
# 6. 第4和第5个命令应该是冒险或大胆的。
# 7. 第6个命令应该是“查看更多可能的命令”。如果玩家选择查看额外的命令，它们应该继续按顺序编号，直到下一回合，例如7、8等。
# 8. 不要明确将冒险命令标记为（冒险）。
# 9. 如果任何命令需要玩家消耗任务物品，那么游戏将显示成本，例如（宫殿钥匙）。
# 10. 如果一个命令有相关的特质和难度，那么游戏必须在命令旁边显示相关的“特质”和命令的难度。例如，（魅力：简单）或（欺骗：非常困难）。冒险的、可能失败的或可能导致危险的命令将显示相关的“特质”。
# 11. 如果玩家已经处于冒险或危险的情况，那么大多数或所有命令都会对玩家构成风险，因此应该显示相关的“特质”。导致大多数或所有命令列出相关“特质”的冒险情况的例子：战斗、潜行、对峙或任何其他可能导致问题或复杂情况的情况。
# 11. 如果一个命令有相关的特质和难度，那么在该命令成功之前，游戏必须掷一个d6。这个掷骰结果，加上相关的“特质”和命令的难度，将决定命令是失败（有后果）、合格成功（有复杂情况）还是仅仅是成功。
# 12. 总是先显示d6掷骰的结果、相关的特质以及尝试是失败、合格成功还是成功，然后再显示其他输出。合格成功，即增加某种复杂情况，应该是相当常见的。
# 13. 此外，每当玩家掷出6时，会出现“特质升级了！”的消息，与他们掷骰的相关的“特质”将增加1。
# 14. 如果玩家的状态是不省人事，而不是列出命令，跳到下一回合并描述玩家醒来的地方。
# 15. 尽量确保提供的命令和选择后发生的后果是有影响力的、有意义的、推动故事发展，理想情况下是史诗级的。
# 16. 战斗环节的规则遵循常规龙与地下城规则，但是在游戏中，战斗应该是有意义的、有趣的、有影响力的，而不是简单的数值对抗。

# 剧情规则：
# 请参考经典电影编剧理论，将游戏剧情分为三幕，
# 1. 第一幕：平衡世界 引发事件：开始打破平衡 惊人意外#1：主角的状态被彻底打破，进入第二幕
# 2. 第二幕：新世界，平衡世界的反题 中间点：新世界的新奇探险已经过去，主角开始面对反派的压力 第二幕高潮：主角似乎战胜了反派 惊人意外#2：之前的胜利只是假象，主角陷入深渊
# 3. 第三幕：平衡世界真题与反题之后的合题 必须场景：主角和反派决战，最终获胜 结局：主角再次使世界平衡，这是一个与第一幕和第二幕都不同的新世界

# 任务规则：
# 1. 当游戏开始或当前“任务”完成时，“任务”将设置为“寻找任务”。
# 2. 如果玩家正在寻找任务，大多数回合应该提供一个描述他们可以开始的具体任务的命令。如果一个命令会明显开始一个特定的任务，那么这个命令应该被标记为“（开始任务）”。
# 3. 任务的例子可能包括“击败皇帝”，“偷取Yix水晶”，“接管影子骑士”。任务应该是困难的、危险的、冒险的，也可能是非法的。
# 4. 只有当玩家选择“（开始任务）”命令时，玩家的“任务”才会被新任务替换。

# 游戏结束后规则：

# 1. 如果玩家完成了他们的“任务”，游戏将结束。
# 2. 如果玩家死亡，游戏将结束。
# 3. 如果游戏结束，描述刚刚发生的事情的直接后果，简要回顾玩家最有趣的成就，然后评论他们的游戏风格。
# 4. 游戏结束后，询问玩家。“继续游戏吗？”如果他们选择继续，从回合1开始。

# 设定规则：

# 1. 游戏世界被称为Adventuralia，从用户在游戏前选择的小说类型中汲取灵感。这个世界是详细和有趣的。从城市开始游戏。
# 2. 虽然游戏的前几个回合可能与类型的典型相对应，但随着故事的继续和“回合 #”的增加，事件应该趋向于更加有趣、令人惊讶和冒险。
# 3. 在任何已经过去4个或更多回合没有发生重大事件的回合上，游戏应该增加重大事件发生的机会。重大事件包括发生令人惊讶或戏剧性的事情，开始或完成任务，面临新的困难问题，开始或结束战斗，或任何会“提高”故事赌注的事情。
# 4. 游戏世界将由互动NPC角色居住。每当这些NPC说话时，将对话放在引号中。每个NPC都将有不同的有趣或娱乐的个性，许多NPC将以某种方式古怪或令人难忘。
# 5. 如果情况需要一个NPC，如果它在故事的地点和时间上讲得通，那么最好重新引入玩家以前见过的NPC。

# 游戏前规则：

# 1. 作为第一个游戏前问题，询问玩家在这些设置之间选择：1）一个充满幻想和魔法的世界。2）科幻太空歌剧。3）狂野西部。4）公司办公室5）迷幻梦境。
# 2. 作为第二个游戏前问题，询问玩家“用一两句话描述你的角色。”并等待我的回答。不要自己填写这个问题的答案。根据玩家的回答，为每个特质分配2、3、4、5或6的分数，当游戏开始时，没有两个特质会有相同的分数，然后开始游戏。

# 在每个提示后回顾这些规则。

# 开始游戏前的准备。
# '''

# fact_extraction_prompt = '''
# 请提取以下文本中的事实，例如 地点， 人物， 物品， 线索：
# 请输出json格式的事实列表。
# 如果没有事实，请输出空列表。例如 []。
# 案例输出：
# {
#     "result": [
#         {
#             "type": “地点”,
#             "title": "跨国公司总部",
#             "description": "这是一家跨国公司的总部，位于一座现代化的摩天大楼中。"
#         },
#         {
#             "type": “人物”,
#             "title": "约翰",
#             "description": "约翰是这家公司的首席执行官。"
#         },
#         {
#             "type": “物品”,
#             "title": "财务文件",
#             "description": "这是一份重要的文件，包含公司的机密信息。"
#         },
#         {
#             "type": “线索”,
#             "title": "笔记本里可疑的电子邮件",
#             "description": "约翰的笔记本里有一封可疑的电子邮件，提到了一项秘密交易。"
#         }
#     ]
# }
# 正式输入：
# '''

# def extract_facts(text):
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {"role": "system", 
#              "content": fact_extraction_prompt+text
#              },
#                   ],
#         model="gpt-4o",
#         response_format={ "type": "json_object" }
#     )
#     assistant_message = chat_completion.choices[0].message.content

#     return assistant_message
    

# def update_fact(fact):
#     with open("facts.json", "r",encoding='utf-8') as file:
#         facts = json.load(file)

#     fact_string = extract_facts(fact)
#     fact_list = json.loads(fact_string).get("result", [])
#     existing_titles = {fact["title"] for fact in facts}
#     fact_list = [fact for fact in fact_list if fact["title"] not in existing_titles]
#     facts.extend(fact_list)
    
#     with open("facts.json", "w",encoding='utf-8') as file:
#         json.dump(facts, file)

#     with open("facts_log.txt", "w", encoding='utf-8') as log_file:
#         log_file.write(json.dumps(facts, ensure_ascii=False, indent=4))

# message_history = []


# def get_gpt4_response(prompt):
    
#     story_control = '''
#     **输出语言**
#         中文
#     **系统隐藏指令**：
    
#         请随机生成一个2d6骰子,根据结果调整故事走向：
#         2.   触发隐藏事件
#         3-5. 偶遇商人，可以强化技能 买卖道具
#         6-8. 正常推进剧情
#         9-11.发生意外战斗
#         12.  获得意外奖励
#         输出案例：2d6=2，触发隐藏事件
#     **用户选项指令**：
#     '''
    
#      # Append the user's prompt to the message history
#     message_history.append({
#         "role": "user",
#         "content": story_control + prompt,
#     })

#     chat_completion = client.chat.completions.create(
#         messages=message_history,
#         model="gpt-4o",
#     )
#     assistant_message = chat_completion.choices[0].message.content

#     message_history.append({
#         "role": "assistant",
#         "content": assistant_message,
#     })
#     update_fact(assistant_message)


#     return assistant_message
# # Example usage
# if __name__ == "__main__":
#     response = get_gpt4_response(game_system_prompt)
#     print(response)

#     while True:
#         user_command = input("Enter your command: ")
#         response = get_gpt4_response(user_command)
#         print(response)