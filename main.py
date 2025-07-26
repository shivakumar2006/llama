from llama_cpp import Llama 

def format_prompt(prompt): 
    return f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{prompt} [/INST]"

def main(): 
    print("🔥 llama project started successfully")

    model_path = "models/llama-2-7b-chat.Q4_K_S.gguf"
    llm = Llama(
        model_path=model_path,
        n_ctx=2048, #context length
        n_threads=4, #tune based on CPU
        n_gpu_layers=0, #optional, if you have GPU
        verbose=False
    )

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chat")
            break

        full_prompt = format_prompt(prompt)
        print(f"Sending prompt:\n{full_prompt}")

        response = llm(
            full_prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>"]
        )

        answer = response["choices"][0]["text"].strip()
        print("LLAMA:", answer)

if __name__ == "__main__":
    main()




# from llama_cpp import Llama

# def main():
#     print("🔥 llama project started successfully")

#     # set model path 
#     model_path = "models/llama-2-7b-chat.Q4_K_S.gguf"
#     llm = Llama(model_path=model_path)

#     #chat loop 
#     while True : 
#         prompt = input("You: ")
#         if prompt.lower() in ["exit", "quit"]:
#             print("Exisitn chat")
#             break 

#         full_prompt = f"Q: {prompt}\nA:"
#         print(f"Sending prompt {full_prompt}")
#         response = llm(full_prompt) 

#         # Print response 
#         answer = response["choices"][0]["text"].strip()
#         print("LLAMA:", answer)

#     # # Load the llama model 
#     # print("Loading model")
#     # llm = Llama(model_path=model_path)

#     # #send prompt 
#     # prompt = "Q: What is the capital of india?\nA:"
#     # print(f"sending prompt: {prompt}")
#     # response = llm(prompt)

#     # #print response 
#     # print("resposne received")
#     # print(response["choices"][0]["text"])

# if __name__ == "__main__":
#     main()