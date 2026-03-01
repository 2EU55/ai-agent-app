import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv(dotenv_path="AI_Intern_Bootcamp/.env")

if os.environ.get("SILICONFLOW_API_KEY"):
    client = OpenAI(
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1"
    )
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    client = OpenAI(api_key="your-api-key")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "搜索指定主题的新闻",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "新闻主题，如AI、科技、体育"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "计算BMI指数",
            "parameters": {
                "type": "object",
                "properties": {
                    "height": {
                        "type": "number",
                        "description": "身高（米）"
                    },
                    "weight": {
                        "type": "number",
                        "description": "体重（千克）"
                    }
                },
                "required": ["height", "weight"]
            }
        }
    }
]

def get_weather(city):
    """模拟天气查询函数"""
    weather_data = {
        "北京": {"weather": "晴", "temp": "25°C"},
        "上海": {"weather": "雨", "temp": "18°C"},
        "深圳": {"weather": "多云", "temp": "28°C"}
    }
    return weather_data.get(city, {"weather": "未知", "temp": "?"})


def search_news(topic):
    """模拟新闻搜索函数"""
    news_data = {
        "AI": ["OpenAI发布新模型", "AI监管法案通过", "ChatGPT用户超1亿"],
        "科技": ["苹果发布新品", "特斯拉自动驾驶更新", "芯片短缺问题缓解"],
        "体育": ["世界杯冠军产生", "NBA交易截止日", "马拉松世界纪录"]
    }
    return news_data.get(topic, ["未找到相关新闻"])


def calculate_bmi(height, weight):
    """计算BMI函数"""
    try:
        bmi = weight / (height * height)
        return json.dumps({"bmi": round(bmi, 2), "status": "正常" if 18.5 <= bmi <= 24.9 else "异常"})
    except Exception as e:
        return str(e)


messages = [
    {"role": "system", "content": "你是一个助手，会调用函数来回答问题。"}
]

user_input = "我身高1.75米，体重70公斤，BMI是多少？"
messages.append({"role": "user", "content": user_input})

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message
print(f"AI回复: {response_message.content}")
print(f"工具调用: {response_message.tool_calls}")

if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        print(f"\n调用函数: {function_name}")
        print(f"参数: {function_args}")

        if function_name == "get_weather":
            result = get_weather(function_args.get("city", ""))
        elif function_name == "search_news":
            result = search_news(function_args.get("topic", ""))
        elif function_name == "calculate_bmi":
            result = calculate_bmi(function_args.get("height"), function_args.get("weight"))
        else:
            result = "未知函数"

        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    final_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages
    )
    print(f"\n最终回复: {final_response.choices[0].message.content}")
