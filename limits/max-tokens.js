const { spawn } = require("child_process");

const pythonCode = `
MAX_TOKENS = 30000

def estimate_tokens(text):
    return (len(text) + 3) // 4

def count_tokens(context):
    return sum(estimate_tokens(x) for x in context)

print(count_tokens(["Hello from Python!"]))
`;

const python = spawn("python", ["-c", pythonCode]);

python.stdout.on("data", (data) => {
    console.log("Python:", data.toString());
});

python.stderr.on("data", (data) => {
    console.error("Python error:", data.toString());
});