// import things
import dotenv from "dotenv";
import { Configuration, OpenAIApi } from "openai";
import * as math from "mathjs";
import { createClient } from '@supabase/supabase-js'

// configure things
dotenv.config();
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY)

// create sample context and query data
const prompt_data = ["light"];
const context_data = [  
  "Candles flicker, casting dancing shadows on the walls.",
  "Bare feet sink into soft, cool sand at the water's edge.",
  "Laughter of children fills the air, a symphony of pure joy.",
  "Stars twinkle like distant diamonds in the midnight sky.",
  "A kitten's playful antics bring smiles to weary faces.",
  "Steam rises from a cup of hot cocoa on a chilly day.",
  "Leaves rustle in the breeze, nature's own gentle music.",
  "A single, bright balloon floats against a clear blue sky.",
  "The aroma of freshly baked bread wafts from the local bakery.",
  "Gentle waves kiss the shore, a rhythmic lullaby for the soul.",
  "A cozy blanket offers warmth on a cold winter morning.",
  "Dewdrops glisten on blades of grass as dawn breaks.",
  "Cherry blossoms fall like confetti, painting the ground pink.",
  "A kind gesture sparks a chain of smiles among strangers.",
  "The distant rumble of thunder adds drama to the stormy sky.",
  "A flickering campfire mesmerizes with its dance of flames.",
  "A rainbow arches gracefully, a bridge of vibrant colors.",
  "The scent of blooming flowers announces the arrival of spring.",
  "Autumn leaves crunch underfoot, a chorus of nature's footsteps.",
  "A baby's laughter rings out, infectious and heartwarming.",
];

// get embeddings for context and query data, map it to the text
//      api request for context data
const context_data_api_response = await openai.createEmbedding({
  model: "text-embedding-ada-002",
  input: context_data,
});

//     extract embeddings from response into map
const embeddingMap = {};
context_data_api_response.data.data.forEach((item) => {
  embeddingMap[item.index] = item.embedding;
});

//      map embeddings to the context data
const context_data_with_embeddings = context_data.map((text, index) => ({
  context: text,
  embedding: embeddingMap[index],
}));

const savetosupbase = await supabase.from('documents').insert(context_data_with_embeddings)

//      api request for prompt data
const prompt_data_api_response = await openai.createEmbedding({
  model: "text-embedding-ada-002",
  input: prompt_data,
});

//      extract embedding
const prompt_data_embedding = prompt_data_api_response.data.data[0].embedding


// do a similarity search on supabase
const search_results = await supabase.rpc('match_documents', {
  query_embedding: prompt_data_embedding, 
  similarity_threshold: 0.78, 
  match_count: 3,
});

// Ask GPT3 to explain why how the prompt relates to the top 3 results
//     create a chat prompt with the 3 most similar items in the context data
let chatPrompt = `I've searched for the term "${prompt_data[0]}" and got the following as the top 3 results:\n`;
search_results.data.forEach((item, index) => {
  chatPrompt += `\n${index + 1}. ${item.context}\n`;
});
chatPrompt += `First repeat the term I searched for and the statements that were returned as top results. Next, concisely explain how the term I searched for relates to each statement.`

//    set up the messages for GPT
const messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": chatPrompt}]

//    query the API
const chatResponse = await openai.createChatCompletion({
  model: "gpt-3.5-turbo",
  messages: messages
});

//    print the final result
console.log(chatResponse.data.choices[0].message.content);