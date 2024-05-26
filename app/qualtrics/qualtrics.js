// Description: Qualtrics JavaScript file for the chat history plugin. 
// Add this file to the Qualtrics survey Question that hosts the chatbot to capture the chat history from the chatbot.

Qualtrics.SurveyEngine.addOnload(function() {
    window.addEventListener('message', function(event) {
        if (event.data.type === 'chatHistory') {
            Qualtrics.SurveyEngine.setEmbeddedData('chatHistory', JSON.stringify(event.data.chatHistory));
        }
    });
});

Qualtrics.SurveyEngine.addOnReady(function()
{
	/*Place your JavaScript here to run when the page is fully displayed*/

});

Qualtrics.SurveyEngine.addOnUnload(function()
{
	/*Place your JavaScript here to run when the page is unloaded*/

});