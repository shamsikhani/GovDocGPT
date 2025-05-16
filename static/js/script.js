document.addEventListener('DOMContentLoaded', () => {
    const chatArea = document.getElementById('chat-area');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');

    const addMessage = (text, sender) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        
        // Sanitize text before adding to HTML to prevent XSS
        // A simple way for now, for production use a robust library like DOMPurify
        const p = document.createElement('p');
        p.textContent = text;
        messageDiv.appendChild(p);
        
        chatArea.appendChild(messageDiv);
        chatArea.scrollTop = chatArea.scrollHeight; // Auto-scroll to the latest message
    };

    const showLoading = (show) => {
        loadingIndicator.style.display = show ? 'block' : 'none';
    };

    const handleSend = async () => {
        const question = userInput.value.trim();
        if (!question) return;

        addMessage(question, 'user');
        userInput.value = '';
        showLoading(true);

        try {
            const response = await fetch('http://localhost:8010/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            showLoading(false);

            if (!response.ok) {
                console.error('API Error Response:', response);
                let errorText = response.statusText;
                try {
                    const errorData = await response.json(); // Try to parse as JSON first
                    console.error('API Error Data (JSON):', errorData);
                    errorText = errorData.detail || errorData.message || response.statusText;
                    addMessage(`Error: ${errorText} (Status: ${response.status})`, 'bot');
                } catch (e) {
                    // If not JSON, try to read as text
                    errorText = await response.text();
                    console.error('API Error Data (Text):', errorText);
                    addMessage(`Error: ${errorText.substring(0, 100)}... (Status: ${response.status})`, 'bot');
                }
                return;
            }

            const data = await response.json();
            addMessage(data.answer, 'bot');

        } catch (error) {
            showLoading(false);
            console.error('Fetch Exception:', error);
            console.error('Error Name:', error.name);
            console.error('Error Message:', error.message);
            console.error('Error Stack:', error.stack);
            addMessage(`Sorry, I encountered a problem connecting to the server. (${error.name || 'Unknown Error'})`, 'bot');
        }
    };

    sendButton.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            handleSend();
        }
    });
});
