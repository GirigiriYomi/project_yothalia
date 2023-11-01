import replicate
import os


os.environ["REPLICATE_API_TOKEN"] = "r8_MQ7RVwv19k2d4BuJ28mbIJZ6ZGHgMIe0cZUoP"

output = replicate.run(
    "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    input={"prompt": "say something"}
)
# The replicate/llama-2-70b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
for item in output:
    # https://replicate.com/replicate/llama-2-70b-chat/versions/2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf/api#output-schema
    print(item)