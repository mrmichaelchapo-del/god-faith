const { spawn } = require("child_process");

const pythonCode = `
MAX_TOKENS = 20_000

def estimate_tokens(text):
    # Approximate token count
    return (len(text) + 3) // 4

def add_message(context, message):
    context.append(message)

    # Remove oldest messages until we're under 20K tokens
    while sum(estimate_tokens(x) for x in context) > MAX_TOKENS:
        context.pop(0)

    return context

def token_count(context):
    return sum(estimate_tokens(x) for x in context)

context = []

# Messages
context = add_message(context, "Hello!")
context = add_message(context, "How are you?")
context = add_message(context, "Tell me about JavaScript.")

print("Messages:", len(context))
print("Estimated tokens:", token_count(context))
print("Maximum tokens:", MAX_TOKENS)
`;

const python = spawn("python", ["-c", pythonCode]);

python.stdout.on("data", (data) => {
    console.log(data.toString());
});

python.stderr.on("data", (data) => {
    console.error("Python error:", data.toString());
});