**What did you learn from implementing a multi-agent workflow?**
I learned about how to use two agents together to arrive at a better solution. You need to split up the descriptions
so that the agents will focus only on their task in order to get the best collaborative result. Additionally, I learned
how to use tool calling to improve accuracy of agent responses.

**Challenges faced and how you addressed them.**
It was challenging coming up with prompts for the agents to use. Specifically, it was difficult to get the reviewer
to present its results in the format I wanted while preserving the quality of its review. It took some manipulation of the wording of the prompt to  get the reviewer to really detail its thought process and be thorough with its work. Additionally, I had some difficulty with how the model was formatting test in its responses, which I resolved by modifying the prompt to address this.

**Any creative ideas, variations, or design choices (e.g., persona roles, prompt design).**
I don't believe so. I think I mostly followed the instructions on what to do. I did get a little creative with
how I told it to list differences but that was it.

**What worked well**
The tool calling was pretty easy and seamless to set up. The existing infrastructure made it really easy to get the 
reviewer to actually use the internet to validate the planner's suggestions, and it resulted in some very good 
suggestions and corrections from the reviewer.

**Room for improvement**
One way I think the application could be improved is in the end result. Its a little unintuitive how it returns the 
reviews review then give the raw plan as an attachment. It would be nice if the reviewers changes could be sent back to 
the planner and used to create a new, revised version of the itinerary, and give the original and delta list as 
attachments to that.

**Note any external tools or GenAI assistance used and why.**
- Copilot was active while completing the assignment.
