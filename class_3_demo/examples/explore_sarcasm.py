from transformers import AutoTokenizer, AutoModelWithLMHead, T5Tokenizer


def eval_conversation(text, tokenizer, model):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

    output = model.generate(input_ids=input_ids, max_length=3)

    dec = [tokenizer.decode(ids) for ids in output]

    label = dec[0]

    return label


def main():
    # 실행을 하면.. 아마도 사전훈련된 모델을 다운로드.
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")


    # For similarity with the training dataset we should replace users mentions in twits for @USER token and urls for URL token.

    twit1 = "Trump just suspended the visa program that allowed me to move to the US to start @USER!" +\
    " Unfortunately, I won’t be able to vote in a few months but if you can, please vote him out, " +\
    "he's destroying what made America great in so many different ways!"

    twit2 = "@USER @USER @USER We have far more cases than any other country, " +\
    "so leaving remote workers in would be disastrous. Makes Trump sense."

    twit3 = "My worry is that i wouldn’t be surprised if half the country actually agrees with this move..."

    me = "Trump doing so??? It must be a mistake... XDDD"

    conversation = twit1 + twit2

    print(eval_conversation(conversation, tokenizer, model))  # Output: 'derison'

    conversation = twit1 + twit3

    print(eval_conversation(conversation, tokenizer, model)) # Output: 'normal'

    conversation = twit1 + me

    print(eval_conversation(conversation, tokenizer, model))  # Output: 'derison'


if __name__ == '__main__':
    main()