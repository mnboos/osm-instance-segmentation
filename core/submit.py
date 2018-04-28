import os
import sys
import crowdai


if __name__ == "__main__":
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    if not api_key:
        raise RuntimeError("API key is required")

    challenge = crowdai.Challenge("crowdAIMappingChallenge", api_key)
    predictions_path = os.path.join(os.getcwd(), "predictions.json")
    assert os.path.isfile(predictions_path)
    result = challenge.submit(predictions_path)
    print(result)
