def transcript_chat_completion(client, transcript, user_question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f'Use this transcript or transcripts to answer any user questions, citing specific quotes: {transcript}'
            },
            {
                "role": "user",
                "content": user_question,
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def query_vector_database(docsearch, user_question):
    relevant_docs = docsearch.similarity_search(user_question)
    relevant_transcripts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevant_docs[:3]])
    return relevant_transcripts