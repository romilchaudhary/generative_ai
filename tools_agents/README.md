# Tools and Agents in Generative AI

## Tools

In the context of Generative AI, tools are predefined functions or interfaces that allow AI models, particularly Large Language Models (LLMs), to interact with external systems, perform computations, or access data beyond their training knowledge. Tools enable LLMs to extend their capabilities by calling APIs, querying databases, running code, or executing specific tasks.

Tools act as bridges between the AI's reasoning and the real world, allowing for more dynamic and capable applications. They are typically defined with clear inputs, outputs, and descriptions to guide the LLM on when and how to use them.

Examples of tools include:
- Web search tools (e.g., searching the internet for current information)
- Calculator tools (e.g., performing mathematical computations)
- Database query tools (e.g., retrieving data from a database)
- Custom API callers (e.g., integrating with third-party services like weather APIs or email services)

When should you use Tools?
    -✅ Math
    -✅ Live data
    -✅ APIs
    -✅ Databases

## Agents

Agents in Generative AI are autonomous systems that combine LLMs with a set of tools to perform complex, multi-step tasks. An agent uses the LLM's reasoning abilities to decide which tools to use, in what sequence, and how to interpret the results to achieve a goal. Agents can handle dynamic environments, adapt to new information, and execute actions iteratively without human intervention for each step.

Agents typically follow a cycle of:
1. Observing the current state or user input
2. Reasoning about what action to take
3. Selecting and using appropriate tools
4. Processing the tool outputs
5. Updating their state or providing a response

    An agent is an AI that:
        1. Understands the goal
        2.  Decides what to do
        3.  Uses tools if needed
        4.  Repeats until goal is done
    Agents are: LLM + reasoning + tool usage

Types of agents include:
- **ReAct Agents**: Reason and act in a loop, alternating between thought and action
- **Tool-Calling Agents**: Focus on selecting and executing tools based on LLM decisions
- **Conversational Agents**: Maintain context in dialogues and use tools to enhance responses
- **Multi-Agent Systems**: Multiple agents collaborating on tasks

Agents are particularly useful for applications requiring planning, decision-making, and interaction with external resources, such as virtual assistants, automated workflows, and intelligent automation systems.

# 🧠 Agent workflow (visual)
User Question
      ↓
Think 🤔
      ↓
Choose Tool 🔧
      ↓
Use Tool
      ↓
Observe Result 👀
      ↓
Answer

This loop is called ReAct (Reason + Act).