# LLM RAG Experiments

Playing around with Retrieval Augmented Generation.

## Next Steps

- [x] Go through [first tutorial](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline)
- [x] Try this with my own dataset (e.g expenses csv)
- [ ] Repeat with my own dataset but this time:
    - [ ] See if there's a converter for CSV to Haystack Document
    - [ ] Split and clean the document
    - [ ] Then see if it provides better answers
- [ ] Understand the pipeline connections (draw a diagram)
- [ ] Make it interactive, with the abiliy to add documents and add questions about newly added documents
- [ ] What are the different configurations and what do they mean?

## Questions

- In Lesson 2, Why did we initialise the InMemoryDocumentStore with `BM25Plus` and use `InMemoryBM25Retriever`?

## Notes

Actually, Hugging Face gave a funny answer in lesson 3. Just noting it here.

** Which of the following recipes would be meet with the most disapproval from Ron Swanson? Explain why in Ron Swanson's style of speech**

> Ron Swanson would most likely disapprove of the vegan persimmon flan recipe due to its use of non-traditional ingredients such as persimmon pulp, agar agar, and coconut creme. 
> 
> In Ron Swanson's style of speech, he might say, "This flan is an abomination! Persimmon pulp in a flan? 
> That's like putting a turkey in a lasagna! It's an affront to all that is holy and delicious in this world. 
> 
> I'll have my men round up every copy of this recipe and burn it in a ceremonial bonfire. From this day forward, flan will be made the way it was meant to be made: with eggs, sugar, and a healthy dose of manliness."