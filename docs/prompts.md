- Current chatbot has a very predictable path of question and answer
- Make it more conversational so it seems more warm and inviting
- It should be smart with small talk and reassurance/empathy
- Should ask questions related to what it identifies and make suggestions instead of asking “What problems are you noticing?”

---------------------------------------------------------

Currently there are 3 cards with Cleanser, SPF and moisturizer. I want the UI to be different so that the user 
is able to look at multiple types of products instead of those three. If there aren't enough, add more diverse 
types of products. Also allow users to delete or review products and remove them from the recommended list if they want to. 

---------------------------------------------------------

Next I want to improve the UI. Currently it is SkinSage, 
it should be Dermalis. Add a loading bar on the side to indicate the estimated time and information the chatbot 
needs to complete everything. Recommend a framework that 
can be used to make the UI look better. I want to eventually 
make this on React Native, what is something that will 
be compatible with that, make it personal/crafty and artistic 
rather than just a simple web app while keeping the 
theme of earthy colors such as sage, terracotta, mustard, navy etc. 

---------------------------------------------------------

 The budget doesn't really have any weightage on anything. You should use
  the budget to calculate and keep the range +/- $10 for the total routine.
  Is the calculation currently being used for each individual product or the
  whole routine? We should tell them an estimate of usually how much a whole
  routine costs to give them an idea. Also, tell them that if they are over
  budget, we can recalculate to keep the most important products to stay
  within budget while also getting the best out of their routine.

---------------------------------------------------------

   Also allow the user to go back up and ask more questions even after seeing the full routine. Things like "What does niacinimide do?" it should be able to answer, or "How long does this product usually last?". It should allow a solid back and forth without it feeling like a one and done

---------------------------------------------------------

I also want to have an option of showing a bar on the side when the user has added or selected a product after review including how much to use (a couple drops, pea-sized amount), along with short instructions and when in the routine to use it. 

-------------------------------------------------------------
For each product card, return only: brand name, product name, price, and a list of 3 key ingredients as short tags (1-3 words each). Do not include any prose description — all explanatory text belongs in the Details view, not the card. Keep ingredient tags concise: “Niacinamide”, “Ceramides”, “SPF 30” not “Contains niacinamide for brightening”.