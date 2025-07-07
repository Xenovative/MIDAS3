// DOM Elements - moved inside DOMContentLoaded to ensure elements exist
let messageInput;
let sendButton;
let chatContainer;
let modelSelect;
let refreshModelsButton;
let newChatButton;
let conversationList;
let chatTitleElement;
let chatModelElement;
let editTitleButton;
let deleteConversationButton;
let sidebarToggle;
let sidebar;
let chatPanel;
let chatInfo;
let docUploadInput;
let fileUploadButton;
let uploadStatus;
let attachmentPreview;
let attachmentPreviewFilename;
let removeAttachmentButton;

// Global variables
let currentModel = '';
let currentConversationId = null;
let isGenerating = false;
let eventSource = null;
let messageTracker = new Set(); // Track message IDs to prevent duplicates
let currentAttachmentFilename = null; // Track attached file for the current message draft
let isNewConversation = false; // NEW GLOBAL VARIABLE
let eventListenersAdded = false; // Flag to prevent duplicate event listeners

// Global variables for preferences management
let userPreferences = {};
let availableModels = [];
let availableEmbeddingModels = [];
let workflowModels = []; // Workflow models for image generation

// --- Notification Function ---
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Trigger fade in
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateY(0)';
    }, 10); // Short delay to allow initial styles to apply

    // Set timeout to remove the notification
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
        // Remove element after transition
        setTimeout(() => {
            notification.remove();
        }, 500); // Match transition duration
    }, duration);
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async function() {
    // Initialize DOM elements after the document is loaded
    messageInput = document.getElementById('message-input');
    sendButton = document.getElementById('send-button');
    chatContainer = document.getElementById('chat-container');
    modelSelect = document.getElementById('model-select');
    refreshModelsButton = document.getElementById('refresh-models');
    newChatButton = document.getElementById('new-chat-button');
    conversationList = document.getElementById('conversation-list');
    chatTitleElement = document.getElementById('chat-title');
    chatModelElement = document.getElementById('chat-model');
    editTitleButton = document.getElementById('edit-title-button');
    deleteConversationButton = document.getElementById('delete-conversation-button');
    sidebarToggle = document.getElementById('sidebar-toggle');
    sidebar = document.querySelector('.sidebar');
    chatPanel = document.querySelector('.chat-panel');
    chatInfo = document.querySelector('.chat-info');
    docUploadInput = document.getElementById('doc-upload-input');
    fileUploadButton = document.getElementById('file-upload-button');
    uploadStatus = document.getElementById('upload-status');
    attachmentPreview = document.getElementById('attachment-preview');
    attachmentPreviewFilename = attachmentPreview.querySelector('span:first-child');
    removeAttachmentButton = attachmentPreview.querySelector('.remove-attachment');

    // Event listener for file input change
    docUploadInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        
        if (file) {
            // Show attachment preview
            attachmentPreviewFilename.textContent = file.name;
            currentAttachmentFilename = file.name;
            attachmentPreview.style.display = 'flex';
        } else {
            // Hide attachment preview if no file selected
            attachmentPreview.style.display = 'none';
            currentAttachmentFilename = null;
        }
    });

    // Initialize attachment preview to hidden
    attachmentPreview.style.display = 'none';

    // Event listener for remove attachment button
    removeAttachmentButton.addEventListener('click', removeAttachment);

    // --- Ensure workflow models are loaded before loading models ---
    await loadWorkflowModels();
    await init();
    
    // Initialize preferences button
    initializePreferences();
    
    // Initialize bot management
    initializeBotManagement();
    
    // Initialize sidebar toggle functionality
    initSidebarToggle();

    // Mobile responsiveness enhancements
    setupMobileResponsiveness();
    
    // Mobile header functionality
    setupMobileHeader();

    // Enhanced: Keep mobile chat title and model in sync, and toggle logo visibility
    function updateMobileChatHeader() {
        const desktopTitle = document.getElementById('chat-title');
        const desktopModel = document.getElementById('chat-model');
        const mobileTitle = document.getElementById('mobile-chat-title');
        const mobileModel = document.getElementById('mobile-chat-model');
        const mobileLogo = document.querySelector('.mobile-logo');

        // Determine if a chat is active (not landing page)
        const isChatActive = desktopTitle && desktopTitle.textContent.trim() && desktopTitle.textContent.trim().toLowerCase() !== 'new chat';

        if (mobileTitle) {
            mobileTitle.textContent = desktopTitle ? desktopTitle.textContent : '';
            mobileTitle.style.display = isChatActive ? 'block' : 'none';
        }
        if (mobileModel) {
            mobileModel.textContent = desktopModel ? desktopModel.textContent : '';
            // Always display the model subtitle under the chat title on mobile if chat is active
            mobileModel.style.display = isChatActive ? 'block' : 'none';
        }
        if (mobileLogo) {
            mobileLogo.style.display = isChatActive ? 'none' : 'flex';
        }
    }

    // Observe changes to the chat title and model, and update mobile header
    const chatTitleElem = document.getElementById('chat-title');
    const chatModelElem = document.getElementById('chat-model');
    const chatHeaderObserver = new MutationObserver(updateMobileChatHeader);
    if (chatTitleElem) {
        chatHeaderObserver.observe(chatTitleElem, { childList: true, subtree: true, characterData: true });
    }
    if (chatModelElem) {
        chatHeaderObserver.observe(chatModelElem, { childList: true, subtree: true, characterData: true });
    }

    // Also update on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateMobileChatHeader);
    } else {
        updateMobileChatHeader();
    }

    // Keep mobile chat title in sync with main chat title
    function updateMobileChatTitle() {
        const desktopTitle = document.getElementById('chat-title');
        const mobileTitle = document.getElementById('mobile-chat-title');
        if (desktopTitle && mobileTitle) {
            mobileTitle.textContent = desktopTitle.textContent;
        }
    }

    // Observe changes to the chat title and update mobile title
    const chatTitleObserver = new MutationObserver(updateMobileChatTitle);
    if (chatTitleElem) {
        chatTitleObserver.observe(chatTitleElem, { childList: true, subtree: true, characterData: true });
        updateMobileChatTitle();
    }

    // Also update mobile chat title on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateMobileChatTitle);
    } else {
        updateMobileChatTitle();
    }
});

// Initialize the application
async function init() {
    // Load available models
    await loadModels();
    
    // Load conversations
    await loadConversations();
    
    // Always show landing page when user logs in
    showLandingPage();
    
    // Add event listeners
    addEventListeners();
}

// Theme toggle functionality
function initThemeToggle() {
    // Get theme from user preferences if available, otherwise fallback to localStorage or system preference
    let preferredTheme;
    
    if (typeof userPreferences !== 'undefined' && userPreferences.theme) {
        // Use theme from user preferences
        preferredTheme = userPreferences.theme;
        
        // Handle 'system' theme preference
        if (preferredTheme === 'system') {
            preferredTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
    } else {
        // Fallback to localStorage or system preference
        preferredTheme = localStorage.getItem('theme') || 
                      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    }
    
    // Apply the theme
    setTheme(preferredTheme);
    
    // Update the toggle button icon
    updateThemeIcon(preferredTheme);
    
    // Add event listener to toggle button
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        setTheme(newTheme);
        updateThemeIcon(newTheme);
        localStorage.setItem('theme', newTheme);
        
        // If we have access to the preferences API, update the user preferences
        if (typeof userPreferences !== 'undefined') {
            // Update local preferences object
            userPreferences.theme = newTheme;
            
            // Send to server
            fetch('/api/preferences', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userPreferences)
            }).catch(error => {
                console.error('Error updating theme preference:', error);
            });
        }
    });
    
    // Add listener for system theme changes if user preference is set to 'system'
    if (typeof userPreferences !== 'undefined' && userPreferences.theme === 'system') {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
            const newTheme = e.matches ? 'dark' : 'light';
            setTheme(newTheme);
            updateThemeIcon(newTheme);
        });
    }
}

// Set the theme on the document
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
}

// Update the theme toggle icon
function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    if (theme === 'dark') {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
}

// Load available models and bots
async function loadModels() {
    try {
        showLoading('Loading available models and bots...');
        
        // Fetch models and bots concurrently
        const [modelsResponse, botsResponse] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/bots')
        ]);
        
        const modelsData = await modelsResponse.json();
        const botsData = await botsResponse.json();
        
        console.log('Model data received:', modelsData); // Debug log
        console.log('Bots data received:', botsData); // Debug log
        
        // Clear existing options
        modelSelect.innerHTML = '';
        
        // Add system models
        if (modelsData.status === 'success' && modelsData.models && modelsData.models.length > 0) {
            const systemModelsGroup = document.createElement('optgroup');
            systemModelsGroup.label = '⚙️ System Models';
            
            // Filter models based on visible_models preference if available
            let visibleModels = [];
            if (userPreferences && userPreferences.visible_models && userPreferences.visible_models.length > 0) {
                visibleModels = userPreferences.visible_models;
                console.log('Visible models from preferences:', visibleModels);
            }
            
            // Log all available models from API
            console.log('All models from API:', modelsData.models.map(model => 
                typeof model === 'object' ? model.name : model
            ));
            
            // Use either the filtered list or all models
            const modelsToDisplay = visibleModels.length > 0 
                ? modelsData.models.filter(model => {
                    const modelName = typeof model === 'object' ? model.name || model.id : model;
                    const isVisible = visibleModels.includes(modelName);
                    if (!isVisible && modelName.includes('mistral-small')) {
                        console.log(`Model ${modelName} is in preferences but filtered out`);
                    }
                    return isVisible;
                })
                : modelsData.models;
                
            console.log('Models to display:', modelsToDisplay.map(model => 
                typeof model === 'object' ? model.name : model
            ));
            
            modelsToDisplay.forEach(model => {
                const option = document.createElement('option');
                
                // Handle both string models and object models
                if (typeof model === 'object') {
                    const modelName = model.name || 'Unknown Model';
                    option.value = modelName;
                    
                    // Create display name with parameter size if available
                    let displayName = modelName;
                    if (model.details && model.details.parameter_size) {
                        displayName += ` (${model.details.parameter_size})`;
                    }
                    option.textContent = displayName;
                    
                    // Add tooltip with additional info
                    let tooltip = '';
                    if (model.modified) {
                        tooltip += `Modified: ${model.modified}\n`;
                    }
                    if (model.size) {
                        const sizeGB = (model.size/1024/1024/1024).toFixed(2);
                        tooltip += `Size: ${sizeGB}GB`;
                    }
                    if (tooltip) {
                        option.title = tooltip;
                    }
                } else {
                    // Simple string model
                    option.value = model;
                    option.textContent = model;
                }
                
                systemModelsGroup.appendChild(option);
            });
            
            modelSelect.appendChild(systemModelsGroup);
        }
        
        // Add custom bots as models
        if (botsData.status === 'success' && botsData.bots && botsData.bots.length > 0) {
            const customBotsGroup = document.createElement('optgroup');
            customBotsGroup.label = '🤖 Custom Bots';
            
            botsData.bots.forEach(bot => {
                const option = document.createElement('option');
                option.value = `bot:${bot.id}`;
                option.textContent = bot.name;
                
                // Add tooltip with bot details
                let tooltip = '';
                if (bot.description) {
                    tooltip += `Description: ${bot.description}\n`;
                }
                if (bot.base_model) {
                    tooltip += `Base Model: ${bot.base_model}`;
                }
                if (tooltip) {
                    option.title = tooltip;
                }
                
                customBotsGroup.appendChild(option);
            });
            
            modelSelect.appendChild(customBotsGroup);
        }
        
        // Add workflow models
        if (workflowModels.length > 0) {
            const workflowModelsGroup = document.createElement('optgroup');
            workflowModelsGroup.label = '🖼️ Image Generation';
            
            workflowModels.forEach(wf => {
                const option = document.createElement('option');
                option.value = 'workflow:' + wf;
                option.textContent = wf;
                option.dataset.imageModel = 'true';
                workflowModelsGroup.appendChild(option);
            });
            
            modelSelect.appendChild(workflowModelsGroup);
        }
        
        // Set the first model as current
        if (modelSelect.options.length > 0) {
            currentModel = modelSelect.options[0].value;
            modelSelect.value = currentModel;
            
            addMessage(`${modelSelect.options.length} models loaded`, false, 'system');
        } else {
            addMessage('No models available', false, 'system');
        }
    } catch (error) {
        addMessage('Failed to load models and bots', false, 'system');
        console.error('Model and bot loading error:', error);
    } finally {
        removeLoading();
    }
}

// Fetch workflow files from backend and populate workflowModels array
async function loadWorkflowModels() {
    try {
        const resp = await fetch('/api/workflows');
        const data = await resp.json();
        if (data.status === 'success' && Array.isArray(data.workflows)) {
            workflowModels = data.workflows;
        } else {
            workflowModels = [];
        }
    } catch (err) {
        workflowModels = [];
    }
}

// Load conversations
async function loadConversations(refresh = false) {
    try {
        showLoading('Loading conversations...');
        
        const response = await fetch('/api/conversations');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Sort by updated_at descending
            const sortedConversations = data.conversations.sort((a, b) => 
                new Date(b.updated_at) - new Date(a.updated_at)
            );
            
            // Get most recent 5 conversations
            const recentConversations = sortedConversations.slice(0, 5);
            
            // Clear existing conversations
            conversationList.innerHTML = '';
            
            // Add recent conversations
            recentConversations.forEach(conversation => {
                const item = document.createElement('div');
                item.className = 'conversation-item';
                item.dataset.id = conversation.id;
                
                if (conversation.id == currentConversationId) {
                    item.classList.add('active-conversation');
                }
                
                // Create title element
                const titleEl = document.createElement('div');
                titleEl.className = 'conversation-title';
                titleEl.textContent = conversation.title || 'New Chat';
                
                // Create metadata container
                const metaContainer = document.createElement('div');
                metaContainer.className = 'conversation-meta';
                
                // Create model element
                const modelEl = document.createElement('span');
                modelEl.className = 'conversation-model';
                
                // Check if this is a bot model
                if (conversation.model && conversation.model.startsWith('bot:')) {
                    // Extract bot ID
                    const botId = conversation.model.substring(4);
                    
                    // Set a temporary value
                    modelEl.textContent = 'Bot';
                    
                    // Fetch the bot details to get its name
                    fetch(`/api/bots/${botId}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                // Display the bot name instead of the ID
                                modelEl.textContent = data.bot.name;
                            } else {
                                // Fallback if bot details can't be fetched
                                modelEl.textContent = conversation.model;
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching bot details:', error);
                            modelEl.textContent = conversation.model;
                        });
                } else {
                    // Regular model, display as is (remove any workflow: prefix)
                    modelEl.textContent = conversation.model.replace('workflow:', '');
                }
                
                // Create date element
                const dateEl = document.createElement('span');
                dateEl.className = 'conversation-date';
                dateEl.textContent = new Date(conversation.updated_at).toLocaleString();
                
                // Add elements to container
                metaContainer.appendChild(modelEl);
                metaContainer.appendChild(dateEl);
                
                item.appendChild(titleEl);
                item.appendChild(metaContainer);
                
                // Add buttons container
                const buttonsContainer = document.createElement('div');
                buttonsContainer.className = 'conversation-buttons';
                
                // Add rename button
                const renameButton = document.createElement('button');
                renameButton.className = 'rename-conversation-button';
                renameButton.innerHTML = '<i class="fas fa-edit"></i>';
                renameButton.title = 'Rename';
                renameButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    renameConversation(conversation.id, conversation.title);
                });
                
                // Add delete button
                const deleteButton = document.createElement('button');
                deleteButton.className = 'delete-conversation-button';
                deleteButton.innerHTML = '<i class="fas fa-trash"></i>';
                deleteButton.title = 'Delete';
                deleteButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    currentConversationToDelete = conversation.id;
                    document.getElementById('delete-confirm-modal').style.display = 'flex';
                });
                
                buttonsContainer.appendChild(renameButton);
                buttonsContainer.appendChild(deleteButton);
                item.appendChild(buttonsContainer);
                
                // Add click handler
                item.addEventListener('click', (e) => {
                    // Only load conversation if clicking on the item itself, not buttons
                    if (e.target === item || e.target.classList.contains('conversation-title') || 
                        e.target.classList.contains('conversation-meta')) {
                        loadConversation(conversation.id);
                    }
                });
                
                conversationList.appendChild(item);
            });
            
            // Add 'More' button if there are more conversations
            if (sortedConversations.length > 5) {
                const moreButton = document.createElement('div');
                moreButton.className = 'more-conversations-button';
                moreButton.textContent = '... More Chats';
                moreButton.addEventListener('click', showAllConversations);
                conversationList.appendChild(moreButton);
            }
        } else {
            addMessage(`Error: ${data.message}`, false, 'system');
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
        conversationList.innerHTML = '<div class="empty-message">Failed to load conversations</div>';
    } finally {
        removeLoading();
    }
}

// Load a conversation
async function loadConversation(conversationId) {
    try {
        showLoading('Loading conversation...');
        
        // Clear chat and show loading
        chatContainer.innerHTML = '';
        
        const response = await fetch(`/api/conversations/${conversationId}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            currentConversationId = conversationId;
            
            // Update UI with loaded conversation
            updateChatInfo(data.conversation.title, data.conversation.model);
            
            // Set the current model
            currentModel = data.conversation.model;
            
            // Update model selection in dropdown
            if (modelSelect) {
                // Try to find the model in the dropdown
                let modelFound = false;
                for (let i = 0; i < modelSelect.options.length; i++) {
                    if (modelSelect.options[i].value === data.conversation.model) {
                        modelSelect.selectedIndex = i;
                        modelFound = true;
                        break;
                    }
                }
                
                // If model not found in dropdown (might be a bot that's not loaded yet)
                if (!modelFound) {
                    // Create a temporary option if needed
                    const tempOption = document.createElement('option');
                    tempOption.value = data.conversation.model;
                    
                    // Check if it's a bot model
                    if (data.conversation.model.startsWith('bot:')) {
                        const botId = data.conversation.model.substring(4);
                        tempOption.textContent = `Loading bot...`;
                        
                        // Fetch bot details
                        fetch(`/api/bots/${botId}`)
                            .then(response => response.json())
                            .then(botData => {
                                if (botData.status === 'success') {
                                    tempOption.textContent = botData.bot.name;
                                } else {
                                    tempOption.textContent = data.conversation.model;
                                }
                            })
                            .catch(error => {
                                console.error('Error fetching bot details:', error);
                                tempOption.textContent = data.conversation.model;
                            });
                    } else {
                        tempOption.textContent = data.conversation.model;
                    }
                    
                    modelSelect.appendChild(tempOption);
                    modelSelect.value = data.conversation.model;
                }
            }
            
            // Render messages, passing the message ID from the database
            data.messages.forEach(message => {
                console.log('Loading message from database:', message);
                
                // Handle image data properly
                let processedImages = message.images;
                
                // If images is a string (JSON that wasn't parsed), try to parse it
                if (typeof message.images === 'string') {
                    try {
                        processedImages = JSON.parse(message.images);
                        console.log('Parsed image data from string:', processedImages);
                    } catch (e) {
                        console.error('Failed to parse image data:', e);
                    }
                }
                
                // Log the processed image data
                if (processedImages) {
                    console.log('Processed images:', processedImages);
                }
                
                addMessage(message.content, false, message.role, message.thinking, message.id, message.attachment_filename, processedImages);
            });
            
            // Update active conversation in sidebar
            document.querySelectorAll('.conversation-item').forEach(item => {
                item.classList.remove('active-conversation');
                if (item.dataset.id === conversationId.toString()) {
                    item.classList.add('active-conversation');
                }
            });
        } else {
            // Suppress error if in secret chat mode
            if (!(secretChatMode || currentConversationSecret)) {
                addMessage(`Error: ${data.message}`, false, 'system');
            }
        }
    } catch (error) {
        console.error('Error loading conversation:', error);
        // Suppress error if in secret chat mode
        if (!(secretChatMode || currentConversationSecret)) {
            addMessage('Failed to load conversation', false, 'system');
        }
    } finally {
        removeLoading();
    }
}

// Rename a conversation
async function renameConversation(conversationId, currentTitle) {
    const newTitle = prompt('Enter a new title for this conversation:', currentTitle);
    
    if (!newTitle || newTitle === currentTitle) return;
    
    try {
        // Update the conversation title
        const response = await fetch(`/api/conversations/${conversationId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: newTitle
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            loadConversations(true); // Force refresh
        } else {
            addMessage(`Error: ${data.message}`, false, 'system');
        }
    } catch (error) {
        addMessage('Failed to rename conversation', false, 'system');
        console.error('Error renaming conversation:', error);
    }
}

// Delete a conversation
async function deleteConversation(id) {
    try {
        const response = await fetch(`/api/conversations/${id}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete conversation');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Show loading message
function showLoading(message) {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'system-message loading-message';
    loadingDiv.textContent = message || 'Loading...';
    loadingDiv.id = 'loading-message';
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Remove loading message
function removeLoading() {
    const loadingMessage = document.getElementById('loading-message');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// Loading Animation Control
function showLoading(message = '') {
    const loadingAnimation = document.getElementById('loading-animation');
    const loadingMessage = document.getElementById('loading-message');
    
    if (loadingAnimation) loadingAnimation.style.display = 'block';
    if (loadingMessage) loadingMessage.textContent = message;
}

function removeLoading() {
    const loadingAnimation = document.getElementById('loading-animation');
    const loadingMessage = document.getElementById('loading-message');
    
    if (loadingAnimation) loadingAnimation.style.display = 'none';
    if (loadingMessage) loadingMessage.textContent = '';
}

// Add markdown renderer with thinking process support
function renderMarkdown(text) {
    // Extract thinking process if present
    let thinkingProcess = '';
    let mainText = text;
    const thinkingMatch = text.match(/<think>([\s\S]*?)<\/think>/);
    
    if (thinkingMatch) {
        thinkingProcess = thinkingMatch[1];
        // Remove thinking tags from main text
        mainText = text.replace(/<think>[\s\S]*?<\/think>/g, '');
    }
    
    // Simple markdown conversion for main text
    let renderedText = '';
    
    // Add thinking process first if present
    if (thinkingProcess) {
        renderedText += `<div class="mb-3 p-2 bg-gray-100 text-gray-700 text-sm rounded">
            <details open>
                <summary class="cursor-pointer font-bold">Thinking Process</summary>
                <div class="mt-2 whitespace-pre-wrap thinking-content">${thinkingProcess.replace(/\n/g, '<br>')}</div>
            </details>
        </div>`;
    }
    
    // Then add the main response
    renderedText += `<div class="main-response">${formatMarkdown(mainText)}</div>`;
    
    return renderedText;
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Function to generate a title based on the first message
async function generateConversationTitle(conversationId, message, model) {
    try {
        // Request the server to generate a title based on the first message
        const response = await fetch(`/api/conversations/${conversationId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                generate_title: true
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.status === 'success') {
            const title = data.title;
            
            // Update page title and header
            updatePageTitle(title);
            updateHeaderInfo(title, model);
            
            // Reload conversations to show new title
            loadConversations(true); // Force refresh
            
            return title;
        }
        
        return null;
    } catch (error) {
        console.error('Error generating title:', error);
        return null;
    }
}

// Update the page title with the conversation title
function updatePageTitle(title) {
    document.title = title ? `${title} - MIDAS3.0` : 'MIDAS3.0';
    
    // If there's a header title element, update it too
    if (chatTitleElement) {
        chatTitleElement.textContent = title || 'New Chat';
    }
}

// Update header info with current chat title and model
function updateHeaderInfo(title, model) {
    // Update title
    if (chatTitleElement) {
        chatTitleElement.textContent = title || 'New Chat';
    }
    
    // Update model
    if (chatModelElement && model) {
        // Check if this is a bot model
        if (model.startsWith('bot:')) {
            // Extract bot ID
            const botId = model.substring(4);
            
            // Fetch the bot details to get its name
            fetch(`/api/bots/${botId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // Display the bot name instead of the ID
                        chatModelElement.textContent = data.bot.name;
                    } else {
                        // Fallback if bot details can't be fetched
                        chatModelElement.textContent = model;
                    }
                })
                .catch(error => {
                    console.error('Error fetching bot details:', error);
                    chatModelElement.textContent = model;
                });
        } else {
            // Regular model, display as is (remove any workflow: prefix)
            chatModelElement.textContent = model.replace('workflow:', '');
        }
    } else if (chatModelElement) {
        chatModelElement.textContent = 'Select a model';
    }
}

// Update chat title and model for both desktop and mobile
function updateChatInfo(title, model) {
    // Update desktop elements
    const desktopTitle = document.getElementById('chat-title');
    const desktopModel = document.getElementById('chat-model');
    if (desktopTitle) desktopTitle.textContent = title || 'New Chat';
    if (desktopModel) desktopModel.textContent = model || 'Select a model';
    
    // Update mobile elements
    const mobileTitle = document.getElementById('mobile-chat-title');
    const mobileModel = document.getElementById('mobile-chat-model');
    const mobileChatInfo = document.getElementById('mobile-chat-info');
    if (mobileTitle) mobileTitle.textContent = title || 'New Chat';
    if (mobileModel) mobileModel.textContent = model || 'Select a model';
    if (mobileChatInfo) {
        // Show mobile chat info when a chat is selected
        mobileChatInfo.style.display = title ? 'block' : 'none';
    }
}

// Add event listeners
function addEventListeners() {
    console.log('Adding event listeners to elements:', { sendButton, messageInput });
    
    // Remove any existing event listeners first
    if (sendButton) {
        const newSendButton = sendButton.cloneNode(true);
        sendButton.parentNode.replaceChild(newSendButton, sendButton);
        sendButton = newSendButton;
    }
    
    if (messageInput) {
        const newMessageInput = messageInput.cloneNode(true);
        messageInput.parentNode.replaceChild(newMessageInput, messageInput);
        messageInput = newMessageInput;
    }
    
    // Send button click event
    if (sendButton) {
        sendButton.addEventListener('click', function handleSendClick(e) {
            console.log('Send button clicked');
            sendMessage();
        });
    } else {
        console.error('Send button element not found!');
    }
    
    // Message input Enter key event
    if (messageInput) {
        messageInput.addEventListener('keydown', function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('Enter key pressed, sending message');
                sendMessage();
            }
        });
    } else {
        console.error('Message input element not found!');
    }
    
    refreshModelsButton.addEventListener('click', loadModels);
    newChatButton.addEventListener('click', async () => {
        await createNewConversation();
    });
    deleteConversationButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        if (!currentConversationId) {
            addMessage('No conversation selected', false, 'system');
            return;
        }
        
        currentConversationToDelete = currentConversationId;
        document.getElementById('delete-confirm-modal').style.display = 'flex';
    });
    editTitleButton.addEventListener('click', editConversationTitle);
    modelSelect.addEventListener('change', updateConversationModel);
    messageInput.focus();

    // Add listener for document upload button
    if (fileUploadButton) {
        fileUploadButton.addEventListener('click', () => {
            docUploadInput.click();
        });
    }
    
    // Add listener for file selection change
    if (docUploadInput) {
        docUploadInput.addEventListener('change', () => {
            if (docUploadInput.files.length > 0) {
                handleDocumentUpload();
            }
        });
    }
    
    // Set flag to true to indicate that event listeners have been added
    eventListenersAdded = true;
    console.log('All event listeners added successfully');
}

// Delete the current conversation
async function deleteCurrentConversation() {
    if (!currentConversationId) {
        addMessage('No active conversation to delete', false, 'system');
        return;
    }
    
    // Confirm deletion
    if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
        return;
    }
    
    try {
        // Delete the conversation
        const response = await deleteConversation(currentConversationId);
        
        if (response) {
            // Create a new conversation
            await createNewConversation();
            
            // Reload conversations list
            await loadConversations();
            
            // Show success message
            addMessage('Conversation deleted successfully', false, 'system');
        } else {
            addMessage('Failed to delete conversation', false, 'system');
        }
    } catch (error) {
        console.error('Error deleting conversation:', error);
        addMessage('Error deleting conversation', false, 'system');
    }
}

// Edit conversation title
async function editConversationTitle() {
    if (!currentConversationId) {
        addMessage('No active conversation to rename', false, 'system');
        return;
    }
    
    // Get current title
    const currentTitle = chatTitleElement.textContent;
    
    // Prompt for new title
    const newTitle = prompt('Enter a new title for this conversation:', currentTitle);
    
    // If cancelled or empty, do nothing
    if (!newTitle || newTitle.trim() === '') {
        return;
    }
    
    try {
        // Update the conversation title
        const response = await fetch(`/api/conversations/${currentConversationId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: newTitle.trim()
            })
        });
        
        if (response.ok) {
            // Update page title and header
            updatePageTitle(newTitle.trim());
            updateHeaderInfo(newTitle.trim(), chatModelElement.textContent);
            
            // Reload conversations to show new title
            loadConversations(true); // Force refresh
            
            // Show success message
            addMessage(`Conversation renamed to: ${newTitle.trim()}`, false, 'system');
        } else {
            addMessage('Failed to rename conversation', false, 'system');
        }
    } catch (error) {
        addMessage('Error renaming conversation', false, 'system');
        console.error('Error renaming conversation:', error);
    }
}

// Initialize sidebar toggle functionality
function initSidebarToggle() {
    console.log('Initializing sidebar toggle with button:', sidebarToggle);
    
    // Check if sidebar toggle button exists
    if (!sidebarToggle) {
        console.error('Sidebar toggle button not found!');
        return;
    }
    
    // Check for saved sidebar state
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    console.log('Saved sidebar state - collapsed:', sidebarCollapsed);
    
    // Set the initial toggle icon based on saved state
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    if (toggleIcon && sidebarCollapsed) {
        toggleIcon.src = '/static/assets/xeno.png';
    }
    
    // Apply the saved state
    if (sidebarCollapsed) {
        toggleSidebar(false); // Don't save state again
    }
    
    // Add event listener to toggle button
    sidebarToggle.addEventListener('click', () => {
        console.log('Sidebar toggle clicked');
        toggleSidebar(true); // Save state
    });
    
    console.log('Sidebar toggle initialized successfully');
}

// Toggle sidebar visibility
function toggleSidebar(saveState = true) {
    console.log('Toggling sidebar - elements:', {sidebar, chatPanel, chatInfo});
    
    if (!sidebar || !chatPanel) {
        console.error('Required elements for sidebar toggle not found!');
        return;
    }
    
    const isCollapsed = sidebar.classList.toggle('sidebar-collapsed');
    chatPanel.classList.toggle('sidebar-collapsed');
    
    console.log('Sidebar collapsed state:', isCollapsed);
    
    // Update the toggle button image
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    if (toggleIcon) {
        if (isCollapsed) {
            toggleIcon.src = '/static/assets/xeno.png';
            
            // No longer need to adjust chatInfo position via JS
            // if (chatInfo) chatInfo.style.left = '100px'; 
        } else {
            toggleIcon.src = '/static/assets/bigxeno.png';
            
            // No longer need to adjust chatInfo position via JS
            // if (chatInfo) chatInfo.style.left = '280px';
        }
    }
    
    // Save the state if requested
    if (saveState) {
        localStorage.setItem('sidebarCollapsed', isCollapsed);
        console.log('Saved sidebar state:', isCollapsed);
    }
}

// Initialize delete confirmation modal
const deleteModal = document.getElementById('delete-confirm-modal');
const confirmDeleteBtn = document.getElementById('confirm-delete');
const cancelDeleteBtn = document.getElementById('cancel-delete');
let currentConversationToDelete = null;

// Handle delete button clicks
document.querySelectorAll('[id^="delete-conversation"]').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentConversationToDelete = currentConversationId;
        deleteModal.style.display = 'flex';
    });
});

// Confirm delete
confirmDeleteBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    if (currentConversationToDelete) {
        try {
            await deleteConversation(currentConversationToDelete);
            if (currentConversationId === currentConversationToDelete) {
                currentConversationId = null;
                messageInput.value = '';
                clearChatPanel();
                updateTopBar(); // Refresh top bar
            }
            loadConversations();
        } catch (error) {
            console.error('Error deleting conversation:', error);
        } finally {
            deleteModal.style.display = 'none';
            currentConversationToDelete = null;
        }
    }
});

// Cancel delete
cancelDeleteBtn.addEventListener('click', () => {
    deleteModal.style.display = 'none';
    currentConversationToDelete = null;
});

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === deleteModal) {
        deleteModal.style.display = 'none';
        currentConversationToDelete = null;
    }
});

// Clear chat panel
function clearChatPanel() {
    chatContainer.innerHTML = '';
    messageHistory = [];
    lastMessageId = 0;
}

// Update top bar
function updateTopBar() {
    updatePageTitle('');
    updateHeaderInfo('', '');
}

// Update conversation model
async function updateConversationModel() {
    const newModel = modelSelect.value;
    currentModel = newModel;
    
    // Check if this is a bot model
    if (newModel.startsWith('bot:')) {
        // Extract bot ID
        const botId = newModel.substring(4);
        
        // Set a temporary loading indicator
        chatModelElement.textContent = 'Loading bot...';
        
        // Fetch the bot details to get its name
        try {
            const botResponse = await fetch(`/api/bots/${botId}`);
            const botData = await botResponse.json();
            
            if (botData.status === 'success') {
                // Display the bot name instead of the ID without the "Bot:" prefix
                chatModelElement.textContent = botData.bot.name;
            } else {
                // Fallback if bot details can't be fetched
                chatModelElement.textContent = newModel.replace('bot:', '');
            }
        } catch (error) {
            console.error('Error fetching bot details:', error);
            chatModelElement.textContent = newModel.replace('bot:', '');
        }
    } else {
        // Update UI immediately for regular models (remove any workflow: prefix)
        chatModelElement.textContent = newModel.replace('workflow:', '');
    }
    
    // If there's an active conversation, update it in the database
    if (currentConversationId) {
        try {
            // Update the conversation model
            const response = await fetch(`/api/conversations/${currentConversationId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: newModel
                })
            });
            
            if (!response.ok) {
                console.error('Failed to update conversation model');
            }
            
            // Reload conversations to reflect the change
            loadConversations(true);
            
        } catch (error) {
            console.error('Error updating conversation model:', error);
        }
    }
    
    // Remove any existing image commands help button
    const existingHelpButton = document.querySelector('.image-commands-help');
    if (existingHelpButton) {
        existingHelpButton.remove();
    }
}

// Show all conversations in a modal
async function showAllConversations() {
    try {
        showLoading('Loading all conversations...');
        
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'conversations-modal';
        
        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.className = 'conversations-modal-content';
        
        // Add close button
        const closeButton = document.createElement('span');
        closeButton.className = 'close-modal';
        closeButton.innerHTML = '&times;';
        closeButton.addEventListener('click', () => modal.remove());
        modalContent.appendChild(closeButton);
        
        // Add title
        const title = document.createElement('h3');
        title.textContent = 'All Conversations';
        modalContent.appendChild(title);
        
        // Fetch all conversations
        const response = await fetch('/api/conversations');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Sort by updated_at descending
            const sortedConversations = data.conversations.sort((a, b) => 
                new Date(b.updated_at) - new Date(a.updated_at)
            );
            
            // Create list container
            const listContainer = document.createElement('div');
            listContainer.className = 'all-conversations-list';
            
            sortedConversations.forEach(conversation => {
                const item = document.createElement('div');
                item.className = 'conversation-item';
                item.dataset.id = conversation.id;
                
                if (conversation.id == currentConversationId) {
                    item.classList.add('active-conversation');
                }
                
                // Create title element
                const titleEl = document.createElement('div');
                titleEl.className = 'conversation-title';
                titleEl.textContent = conversation.title || 'New Chat';
                
                // Create metadata container
                const metaContainer = document.createElement('div');
                metaContainer.className = 'conversation-meta';
                
                // Create model element
                const modelEl = document.createElement('span');
                modelEl.className = 'conversation-model';
                
                // Check if this is a bot model
                if (conversation.model && conversation.model.startsWith('bot:')) {
                    // Extract bot ID
                    const botId = conversation.model.substring(4);
                    
                    // Set a temporary value
                    modelEl.textContent = 'Bot';
                    
                    // Fetch the bot details to get its name
                    fetch(`/api/bots/${botId}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                // Display the bot name instead of the ID
                                modelEl.textContent = data.bot.name;
                            } else {
                                // Fallback if bot details can't be fetched
                                modelEl.textContent = conversation.model;
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching bot details:', error);
                            modelEl.textContent = conversation.model;
                        });
                } else {
                    // Regular model, display as is (remove any workflow: prefix)
                    modelEl.textContent = conversation.model.replace('workflow:', '');
                }
                
                // Create date element
                const dateEl = document.createElement('span');
                dateEl.className = 'conversation-date';
                dateEl.textContent = new Date(conversation.updated_at).toLocaleString();
                
                // Add elements to container
                metaContainer.appendChild(modelEl);
                metaContainer.appendChild(dateEl);
                
                item.appendChild(titleEl);
                item.appendChild(metaContainer);
                
                // Add buttons container
                const buttonsContainer = document.createElement('div');
                buttonsContainer.className = 'conversation-buttons';
                
                // Add rename button
                const renameButton = document.createElement('button');
                renameButton.className = 'rename-conversation-button';
                renameButton.innerHTML = '<i class="fas fa-edit"></i>';
                renameButton.title = 'Rename';
                renameButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    renameConversation(conversation.id, conversation.title);
                });
                
                // Add delete button
                const deleteButton = document.createElement('button');
                deleteButton.className = 'delete-conversation-button';
                deleteButton.innerHTML = '<i class="fas fa-trash"></i>';
                deleteButton.title = 'Delete';
                deleteButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    currentConversationToDelete = conversation.id;
                    document.getElementById('delete-confirm-modal').style.display = 'flex';
                });
                
                buttonsContainer.appendChild(renameButton);
                buttonsContainer.appendChild(deleteButton);
                item.appendChild(buttonsContainer);
                
                // Add click handler
                item.addEventListener('click', (e) => {
                    // Only load conversation if clicking on the item itself, not buttons
                    if (e.target === item || e.target.classList.contains('conversation-title') || 
                        e.target.classList.contains('conversation-meta')) {
                        loadConversation(conversation.id);
                        modal.remove();
                    }
                });
                
                listContainer.appendChild(item);
            });
            
            modalContent.appendChild(listContainer);
        }
        
        modal.appendChild(modalContent);
        document.body.appendChild(modal);
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    } catch (error) {
        console.error('Error loading all conversations:', error);
    } finally {
        removeLoading();
    }
}

// Track messages to prevent duplicates
let lastMessageId = 0;
let messageHistory = [];

function addMessageButtons(messageElement, role) {
    const actionsContainer = document.createElement('div');
    actionsContainer.className = 'message-actions'; // CSS class handles positioning, display, gap, etc.

    // Copy button
    const copyBtn = document.createElement('button');
    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
    copyBtn.title = 'Copy';
    copyBtn.style.cursor = 'pointer';
    copyBtn.addEventListener('click', () => {
        let textToCopy = '';
        if (role === 'assistant') {
            // Try to get text from main-response first, fallback to whole element
            const mainResponse = messageElement.querySelector('.main-response');
            textToCopy = mainResponse ? mainResponse.innerText : messageElement.innerText;
        } else {
             // For user/system messages, get the whole innerText
            textToCopy = messageElement.innerText;
             // Remove the button text if it got included
            const buttonText = actionsContainer.innerText;
            if (textToCopy.endsWith(buttonText)) {
                textToCopy = textToCopy.slice(0, -buttonText.length).trim();
            }
        }
        navigator.clipboard.writeText(textToCopy).then(() => {
             // Show notification instead of changing icon
             showNotification('Copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Copy failed', err);
            showNotification('Copy failed.', 'error');
        });
    });

    // Share button
    const shareBtn = document.createElement('button');
    shareBtn.innerHTML = '<i class="fas fa-share"></i>';
    shareBtn.title = 'Share';
    shareBtn.style.cursor = 'pointer';
    shareBtn.addEventListener('click', () => {
        let textToShare = '';
         if (role === 'assistant') {
            const mainResponse = messageElement.querySelector('.main-response');
            textToShare = mainResponse ? mainResponse.innerText : messageElement.innerText;
        } else {
            textToShare = messageElement.innerText;
            const buttonText = actionsContainer.innerText;
            if (textToShare.endsWith(buttonText)) {
                textToShare = textToShare.slice(0, -buttonText.length).trim();
            }
        }
        if (navigator.share) {
            navigator.share({
                text: textToShare
            }).catch(err => {
                console.error('Share failed', err);
                // Show notification only if share fails and it's not an AbortError
                if (err.name !== 'AbortError') {
                    showNotification('Could not share message.', 'error');
                }
            });
        } else {
            // Fallback for browsers that don't support navigator.share
            navigator.clipboard.writeText(textToShare).then(() => {
                showNotification('Message copied. Share it anywhere!', 'info');
            }).catch(err => {
                console.error('Copy for share failed', err);
                showNotification('Could not copy message.', 'error');
            });
        }
    });

    // Delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
    deleteBtn.title = 'Delete';
    deleteBtn.style.cursor = 'pointer';
    deleteBtn.addEventListener('click', async () => {
        const messageDbId = messageElement.dataset.messageId;

        if (messageDbId) {
            // If we have a database ID, try to delete from backend
            try {
                const response = await fetch(`/api/messages/${messageDbId}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    console.log(`Message ${messageDbId} deleted from backend.`);
                    messageElement.remove(); // Remove from UI on success
                    showNotification('Message deleted.', 'info'); // Show notification on success
                } else {
                    const errorData = await response.json();
                    console.error(`Failed to delete message ${messageDbId} from backend:`, errorData.message);
                    showNotification(`Error deleting message: ${errorData.message}`, 'error');
                }
            } catch (error) {
                console.error(`Network or other error deleting message ${messageDbId}:`, error);
                showNotification('Error deleting message. Check connection.', 'error');
    }
        } else {
            // If no DB ID, just remove from UI (message wasn't saved or ID wasn't retrieved)
            console.log('Removing message from UI only (no DB ID found).');
            messageElement.remove();
            showNotification('Message removed from view.', 'info'); // Notify user it's only UI removal
        }
    });

    actionsContainer.appendChild(copyBtn);
    actionsContainer.appendChild(shareBtn);
    actionsContainer.appendChild(deleteBtn);
    messageElement.appendChild(actionsContainer);
}

// Function to sanitize content by removing HTML tags
function sanitizeContent(content) {
    if (!content) return '';
    return content.replace(/<[^>]*>/g, '');
}

// Function to add a message to the chat
function addMessage(content, isLoading = false, role = 'user', thinking = '', messageId = null, attachmentFilename = null, images = null) { 
    // Sanitize content to remove any HTML tags
    const sanitizedContent = sanitizeContent(content);
    // Skip if this is a duplicate of the last message
    if (messageHistory.length > 0 && 
        messageHistory[messageHistory.length-1].content === sanitizedContent &&
        messageHistory[messageHistory.length-1].isUser === (role === 'user')) {
        return;
    }

    // Use the passed messageId (from DB) if available, otherwise null
    const dbId = messageId;
    // Use a unique DOM ID based on role and DB ID (if available) or timestamp
    const domId = `message-${role}-${dbId || Date.now()}`;

    // Note: messageHistory uses its own internal ID for UI tracking, not necessarily the DB ID
    const historyId = ++lastMessageId;
    messageHistory.push({
        id: historyId, 
        content: sanitizedContent, 
        isUser: role === 'user', 
        type: role, 
        thinking, 
        dbId: dbId,
        attachmentFilename: attachmentFilename,
        images: images
    });

    const messageElement = document.createElement('div');
    messageElement.id = domId; // Use the unique DOM ID
    messageElement.className = `${role}-message`;
    if (dbId) {
        messageElement.dataset.messageId = dbId; // Store the actual DB message ID
    }
    
    if (role === 'assistant') {
        // For assistant messages, always create thinking and response containers
        // Thinking element first (above the response)
        if (thinking) {
            const thinkingElement = document.createElement('details');
            thinkingElement.className = 'thinking';
            thinkingElement.open = userPreferences.show_thinking || false;
            
            const summary = document.createElement('summary');
            summary.textContent = 'Thinking Process';
            thinkingElement.appendChild(summary);
            
            const thinkingContentEl = document.createElement('div');
            thinkingContentEl.className = 'thinking-content';
            thinkingContentEl.innerHTML = renderMarkdown(thinking);
            thinkingElement.appendChild(thinkingContentEl);
            
            messageElement.appendChild(thinkingElement);
        }
        
        // Check if this message has images
        if (images && Array.isArray(images) && images.length > 0 && images[0]) {
            console.log('Rendering message with images:', images);
            console.log('First image type:', typeof images[0]);
            console.log('First image length:', images[0] ? images[0].length : 'null');
            
            // Create a container for the image and buttons
            const imageContainer = document.createElement('div');
            imageContainer.className = 'generated-image-container';
            
            // Add a scaled-down image (max height 350px)
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${images[0]}`; // Use the first image
            img.className = 'generated-image';
            img.alt = 'Generated Image';
            img.style.maxHeight = '350px';
            imageContainer.appendChild(img);
            
            // Create buttons container
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'message-actions image-actions';
            
            // View full image button
            const viewBtn = document.createElement('button');
            viewBtn.innerHTML = '<i class="fas fa-search-plus"></i>';
            viewBtn.title = 'View Full Image';
            viewBtn.addEventListener('click', () => {
                // Open image in new window or tab
                const win = window.open();
                win.document.write(`
                    <html>
                        <head>
                            <title>Full Image</title>
                            <style>
                                body { margin: 0; background: #000; display: flex; align-items: center; justify-content: center; height: 100vh; }
                                img { max-width: 100%; max-height: 100vh; }
                            </style>
                        </head>
                            <body>
                                <img src="data:image/png;base64,${images[0]}" />
                            </body>
                        </html>
                    `);
            });
            actionsContainer.appendChild(viewBtn);
            
            // Download button
            const dlBtn = document.createElement('button');
            dlBtn.innerHTML = '<i class="fas fa-download"></i>';
            dlBtn.title = 'Download Image';
            dlBtn.addEventListener('click', () => {
                // Create an anchor element to download the image
                const a = document.createElement('a');
                a.href = `data:image/png;base64,${images[0]}`;
                a.download = attachmentFilename || 'generated.png';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            });
            actionsContainer.appendChild(dlBtn);
            
            // Add buttons to the container
            imageContainer.appendChild(actionsContainer);
            
            // Add workflow info if available
            if (content.includes('Generated image using workflow:')) {
                const workflowInfo = document.createElement('div');
                workflowInfo.className = 'workflow-info';
                workflowInfo.textContent = content;
                workflowInfo.style.fontSize = '0.8em';
                workflowInfo.style.color = 'var(--text-color-muted)';
                workflowInfo.style.marginTop = '8px';
                imageContainer.appendChild(workflowInfo);
            }
            
            // Add seed information if available
            if (content.includes('Seed:')) {
                const seedInfo = document.createElement('div');
                seedInfo.className = 'seed-info';
                seedInfo.textContent = content;
                seedInfo.style.fontSize = '0.8em';
                seedInfo.style.color = 'var(--text-color-muted)';
                seedInfo.style.marginTop = '8px';
                imageContainer.appendChild(seedInfo);
            }
            
            // Replace loading with the image container
            messageElement.appendChild(imageContainer);
            
            // Enable image overlay
            enableImageOverlay(img);
        } else {
            // Main response container (below thinking)
            const mainResponse = document.createElement('div');
            mainResponse.className = 'main-response';
            
            if (isLoading) {
                const loadingIndicator = document.createElement('div');
                loadingIndicator.className = 'loading-indicator';
                loadingIndicator.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Thinking...';
                mainResponse.appendChild(loadingIndicator);
            } else {
                mainResponse.innerHTML = renderMarkdown(sanitizedContent);
            }
            
            messageElement.appendChild(mainResponse);
        }
    } else if (role === 'user') {
        // For user messages, display content and attachment if present
        messageElement.innerHTML = formatMarkdown(sanitizedContent);
        
        // Add attachment display if there's an attachment filename
        if (attachmentFilename) {
            const attachmentDiv = document.createElement('div');
            attachmentDiv.className = 'message-attachment';
            attachmentDiv.innerHTML = `
                <i class="fas fa-paperclip"></i> 
                <span class="attachment-filename">${attachmentFilename}</span>
            `;
            messageElement.appendChild(attachmentDiv);
        }
    } else {
        // For system messages, just set innerHTML
        messageElement.innerHTML = formatMarkdown(sanitizedContent);
}

    // Add buttons only for user and assistant messages
    if (role !== 'system') {
        addMessageButtons(messageElement, role);
    }

    chatContainer.appendChild(messageElement);
    scrollToBottom();
    
    return messageElement;
}

// Format markdown with common formatting and links
function formatMarkdown(text) {
    if (!text) return '';
    
    // Clean up various answer prefixes
    text = text
        // Handle at the beginning of text
        .replace(/^(\s*)(Answer:|Final Answer:|My answer:|Here's my answer:|The answer is:)(\s*)/i, '')
        // Handle after a newline
        .replace(/\n(\s*)(Answer:|Final Answer:|My answer:|Here's my answer:|The answer is:)(\s*)/gi, '\n');
    
    // Process markdown links [text](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function(match, text, url) {
        // Ensure URL has a protocol
        if (!/^https?:\/\//.test(url)) {
            url = 'http://' + url;
        }
        return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-blue-500 hover:underline">${text}</a>`;
    });
    
    // Process bare URLs - exclude closing parenthesis from URL capture
    text = text.replace(/(https?:\/\/[^\s<]+[^<.,:;"'\]\s\)])/g, function(url) {
        // Skip if already in an anchor tag
        if (/<a\s[^>]*href=["']?[^>]*>.*<\/a>/i.test(url)) {
            return url;
        }
        return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-blue-500 hover:underline">${url}</a>`;
    });
    
    // Basic markdown formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // italic
        .replace(/`([^`]+)`/g, '<code class="bg-gray-100 dark:bg-gray-700 px-1 rounded">$1</code>') // code
        .replace(/\n/g, '<br>')                            // newlines
        .replace(/^##\s+(.*$)/gm, '<h2 class="text-xl font-bold mt-4 mb-2">$1</h2>')  // h2
        .replace(/^###\s+(.*$)/gm, '<h3 class="text-lg font-bold mt-3 mb-1">$1</h3>'); // h3
}

// Helper: Detect if the selected model/bot is image-capable (ComfyUI bot or workflow model)
function isImageModel(selectedModel, botDetails = null) {
    // Workflow models: any model that starts with 'workflow:'
    if (selectedModel.startsWith('workflow:')) return true;
    // For bots: treat any bot whose name or base_model contains 'comfyui' or 'diffusion' as image model
    if (selectedModel.startsWith('bot:') && botDetails) {
        const name = (botDetails.name || '').toLowerCase();
        const base = (botDetails.base_model || '').toLowerCase();
        return name.includes('comfyui') || name.includes('diffusion') || base.includes('comfyui') || base.includes('diffusion');
    }
    return false;
}

// Show quota exceeded modal
function showQuotaExceededModal(quotaInfo) {
    // Ensure quotaInfo is an object
    quotaInfo = quotaInfo || {};
    console.log('Quota info received:', JSON.stringify(quotaInfo, null, 2));
    
    const modal = document.getElementById('quota-exceeded-modal');
    const quotaMessage = document.getElementById('quota-message');
    const dailyQuotaUsage = document.getElementById('daily-quota-usage');
    const monthlyQuotaUsage = document.getElementById('monthly-quota-usage');
    const closeButton = document.getElementById('close-quota-modal');
    const confirmButton = document.getElementById('confirm-quota-modal');
    
    if (!modal || !quotaMessage || !dailyQuotaUsage || !monthlyQuotaUsage) {
        console.error('Quota modal elements not found');
        return;
    }
    
    // Check if we have the full quota info (from our debugging addition)
    const fullQuota = quotaInfo.full_quota || {};
    
    // Extract all possible quota values from the response
    // For daily values
    let dailyUsed = 0;
    let dailyLimit = null;
    
    // Try to get values from full_quota first (most accurate)
    if (fullQuota.messages_used_today !== undefined) {
        dailyUsed = Number(fullQuota.messages_used_today);
    } else if (quotaInfo.daily_used !== undefined) {
        dailyUsed = Number(quotaInfo.daily_used);
    } else if (quotaInfo.messages_used_today !== undefined) {
        dailyUsed = Number(quotaInfo.messages_used_today);
    } else if (quotaInfo.reason === 'daily_limit' && quotaInfo.used !== undefined) {
        dailyUsed = Number(quotaInfo.used);
    }
    
    // Get daily limit
    if (fullQuota.daily_message_limit !== undefined) {
        dailyLimit = Number(fullQuota.daily_message_limit);
    } else if (quotaInfo.reason === 'daily_limit' && quotaInfo.limit !== undefined) {
        dailyLimit = Number(quotaInfo.limit);
    } else if (quotaInfo.daily_limit !== undefined) {
        dailyLimit = Number(quotaInfo.daily_limit);
    } else if (quotaInfo.daily_message_limit !== undefined) {
        dailyLimit = Number(quotaInfo.daily_message_limit);
    }
    
    // For monthly values
    let monthlyUsed = 0;
    let monthlyLimit = null;
    
    // Try to get values from full_quota first (most accurate)
    if (fullQuota.messages_used_month !== undefined) {
        monthlyUsed = Number(fullQuota.messages_used_month);
    } else if (quotaInfo.monthly_used !== undefined) {
        monthlyUsed = Number(quotaInfo.monthly_used);
    } else if (quotaInfo.messages_used_month !== undefined) {
        monthlyUsed = Number(quotaInfo.messages_used_month);
    } else if (quotaInfo.reason === 'monthly_limit' && quotaInfo.used !== undefined) {
        monthlyUsed = Number(quotaInfo.used);
    }
    
    // Get monthly limit
    if (fullQuota.monthly_message_limit !== undefined) {
        monthlyLimit = Number(fullQuota.monthly_message_limit);
    } else if (quotaInfo.reason === 'monthly_limit' && quotaInfo.limit !== undefined) {
        monthlyLimit = Number(quotaInfo.limit);
    } else if (quotaInfo.monthly_limit !== undefined) {
        monthlyLimit = Number(quotaInfo.monthly_limit);
    } else if (quotaInfo.monthly_message_limit !== undefined) {
        monthlyLimit = Number(quotaInfo.monthly_message_limit);
    }
    
    // Handle null/zero values for limits
    if (dailyLimit === 0 || isNaN(dailyLimit) || dailyLimit === null || dailyLimit === undefined) {
        dailyLimit = null; // Set to null for unlimited
    }
    
    if (monthlyLimit === 0 || isNaN(monthlyLimit) || monthlyLimit === null || monthlyLimit === undefined) {
        monthlyLimit = null; // Set to null for unlimited
    }
    
    console.log('Processed values:', {
        dailyUsed, dailyLimit, monthlyUsed, monthlyLimit,
        dailyLimitType: typeof dailyLimit,
        monthlyLimitType: typeof monthlyLimit
    });
    
    // Set quota message based on reason
    if (quotaInfo.reason === 'daily_limit') {
        const limitDisplay = dailyLimit === null ? '∞' : dailyLimit;
        quotaMessage.textContent = `You've reached your daily message limit (${dailyUsed}/${limitDisplay}). Please try again tomorrow.`;
    } else if (quotaInfo.reason === 'monthly_limit') {
        const limitDisplay = monthlyLimit === null ? '∞' : monthlyLimit;
        quotaMessage.textContent = `You've reached your monthly message limit (${monthlyUsed}/${limitDisplay}). Please try again next month.`;
    } else {
        quotaMessage.textContent = 'You have exceeded your message quota.';
    }
    
    // Display the quota values
    // For daily limit
    if (dailyLimit === null) {
        dailyQuotaUsage.textContent = `${dailyUsed}/∞`;
    } else {
        dailyQuotaUsage.textContent = `${dailyUsed}/${dailyLimit}`;
    }
    
    // For monthly limit
    if (monthlyLimit === null) {
        monthlyQuotaUsage.textContent = `${monthlyUsed}/∞`;
    } else {
        monthlyQuotaUsage.textContent = `${monthlyUsed}/${monthlyLimit}`;
    }
    
    // Show the modal
    modal.style.display = 'flex';
    
    // Add event listeners to close the modal
    if (closeButton) {
        closeButton.onclick = () => {
            modal.style.display = 'none';
        };
    }
    
    if (confirmButton) {
        confirmButton.onclick = () => {
            modal.style.display = 'none';
        };
    }
    
    // Close when clicking outside the modal content
    modal.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    // Don't send if empty (unless there's an attachment)
    if (!message && !currentAttachmentFilename) return;
    
    // Create conversation if none exists
    if (!currentConversationId) {
        await createNewConversation();
    }
    
    // Clear input and reset height
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Add user message to UI
    const userMessageElement = addMessage(message, false, 'user', '', null, currentAttachmentFilename);
    
    // Remember attachment filename before clearing
    const sentAttachmentFilename = currentAttachmentFilename;
    
    // Start generating response
    isGenerating = true;
    updateSendButtonState();
    
    // Get the selected model
    const selectedModel = modelSelect.value;
    
    // Bot-specific configurations
    let botDetails = null;
    let modelConfig = { model: selectedModel };
    
    // If this is a bot model, fetch its details
    if (selectedModel.startsWith('bot:')) {
        try {
            const botId = selectedModel.substring(4); // Remove 'bot:' prefix
            const response = await fetch(`/api/bots/${botId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                botDetails = data.bot;
                
                // Extract bot configuration for API request
                modelConfig = {
                    model: botDetails.base_model,
                    system_prompt: botDetails.system_prompt || '',
                    parameters: botDetails.parameters || {},
                    knowledge_files: botDetails.knowledge_files || []
                };
            }
        } catch (error) {
            console.error('Error fetching bot details:', error);
            // Fallback to default model
            modelConfig = { model: selectedModel };
        }
    }
    
    // Track the current image generation request
    let currentImageRequestId = null;
    
    // IMAGE BOT HANDLING
    if (isImageModel(selectedModel, botDetails)) {
        // Add assistant message with loading spinner
        const responseElement = addMessage('', true, 'assistant');
        responseElement.innerHTML = `<div class="image-loading-spinner"><img src="/static/assets/xenoimggen.png" class="xeno-loader" alt="Loading"> MIDAS is doing AI magic...</div>`;
        isGenerating = true;
        startAiMagicDotsAnimation();
        try {
            // First, save the user message to the database
            if (currentConversationId) {
                try {
                    const userMsgResponse = await fetch(`/api/conversations/${currentConversationId}/messages`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            role: 'user',
                            content: message,
                            attachment_filename: sentAttachmentFilename
                        })
                    });
                    
                    const userMsgData = await userMsgResponse.json();
                    if (userMsgData.status !== 'success') {
                        console.error('Failed to save user message:', userMsgData.message);
                    }
                } catch (msgError) {
                    console.error('Error saving user message:', msgError);
                }
            }
            
            // If workflow, send model: selectedModel and conversation ID
            let body = { 
                prompt: message,
                conversation_id: currentConversationId || null
            };
            if (selectedModel.startsWith('workflow:')) {
                body.model = selectedModel;
            }
            // Add a request ID to track this specific generation request
            const requestId = Date.now();
            currentImageRequestId = requestId;
            
            // Set a timeout for the request (5 minutes)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000);
            
            try {
                const resp = await fetch('/api/generate_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(body),
                    signal: controller.signal
                });
                
                // Clear the timeout if the request completes in time
                clearTimeout(timeoutId);
                
                // If another request was made, ignore this response
                if (currentImageRequestId !== requestId) {
                    return;
                }
                
                if (!resp.ok) {
                    throw new Error(`HTTP error! status: ${resp.status}`);
                }
                
                const data = await resp.json();
                
                if (data.status === 'success' && data.image_base64) {
                // Create a container for the image and buttons
                const imageContainer = document.createElement('div');
                imageContainer.className = 'generated-image-container';
                
                // Add a scaled-down image (max height 350px)
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${data.image_base64}`;
                img.className = 'generated-image';
                img.alt = 'Generated Image';
                img.style.maxHeight = '350px';
                imageContainer.appendChild(img);
                
                // Create buttons container
                const actionsContainer = document.createElement('div');
                actionsContainer.className = 'message-actions image-actions';
                
                // View full image button
                const viewBtn = document.createElement('button');
                viewBtn.innerHTML = '<i class="fas fa-search-plus"></i>';
                viewBtn.title = 'View Full Image';
                viewBtn.addEventListener('click', () => {
                    // Open image in new window or tab
                    const win = window.open();
                    win.document.write(`
                        <html>
                            <head>
                                <title>Full Image</title>
                                <style>
                                    body { margin: 0; background: #000; display: flex; align-items: center; justify-content: center; height: 100vh; }
                                    img { max-width: 100%; max-height: 100vh; }
                                </style>
                            </head>
                            <body>
                                <img src="data:image/png;base64,${data.image_base64}" />
                            </body>
                        </html>
                    `);
                });
                actionsContainer.appendChild(viewBtn);
                
                // Download button
                const dlBtn = document.createElement('button');
                dlBtn.innerHTML = '<i class="fas fa-download"></i>';
                dlBtn.title = 'Download Image';
                dlBtn.addEventListener('click', () => {
                    // Create an anchor element to download the image
                    const a = document.createElement('a');
                    a.href = `data:image/png;base64,${data.image_base64}`;
                    a.download = data.filename || 'generated.png';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                });
                actionsContainer.appendChild(dlBtn);
                
                // Add buttons to the container
                imageContainer.appendChild(actionsContainer);
                
                // Add workflow info if available
                if (data.workflow) {
                    const workflowInfo = document.createElement('div');
                    workflowInfo.className = 'workflow-info';
                    workflowInfo.textContent = '';
                    workflowInfo.style.fontSize = '0.8em';
                    workflowInfo.style.color = 'var(--text-color-muted)';
                    workflowInfo.style.marginTop = '8px';
                    imageContainer.appendChild(workflowInfo);
                }
                
                // Add seed information if available
                if (data.seed !== undefined) {
                    const seedInfo = document.createElement('div');
                    seedInfo.className = 'seed-info';
                    seedInfo.textContent = ``;
                    seedInfo.style.fontSize = '0.8em';
                    seedInfo.style.color = 'var(--text-color-muted)';
                    seedInfo.style.marginTop = '8px';
                    imageContainer.appendChild(seedInfo);
                }
                
                // Replace loading with the image container
                responseElement.innerHTML = '';
                responseElement.appendChild(imageContainer);
                
                // Enable image overlay
                enableImageOverlay(img);
                
                // Generate title for new conversations
                if (isNewConversation) {
                    console.log("Generating title for new image conversation:", currentConversationId);
                    try {
                        // Update conversation list to show latest message
                        loadConversations(true);
                        
                        // Generate a title based on the prompt
                        const titleResponse = await fetch(`/api/conversations/${currentConversationId}/generate-title`, {
                            method: 'POST'
                        });
                        const titleData = await titleResponse.json();
                        
                        if (titleData.status === 'success' && titleData.title) {
                            console.log("Image conversation title generated successfully:", titleData.title);
                            // Update the title in the UI
                            updatePageTitle(titleData.title);
                            updateHeaderInfo(titleData.title, selectedModel);
                            
                            // Reload conversations to show new title
                            loadConversations(true);
                        } else {
                            console.error("Image title generation failed:", titleData);
                        }
                    } catch (titleError) {
                        console.error('Error generating title:', titleError);
                    } finally {
                        // Reset the new conversation flag
                        isNewConversation = false;
                    }
                }
            } else {
                responseElement.innerHTML = `<span class="error-message">Image generation failed: ${data.message || 'Unknown error'}</span>`;
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                responseElement.innerHTML = `<span class="error-message">Image generation timed out. The image may still be processing in the background.</span>`;
            } else {
                console.error('Image generation error:', error);
                responseElement.innerHTML = `<span class="error-message">Image generation failed: ${error.message}</span>`;
                }
            }
        } finally {
            isGenerating = false;
            stopAiMagicDotsAnimation();
            currentImageRequestId = null;
        }
        return;
    }
    // END IMAGE BOT HANDLING

    // Start generating response (text, as before)
    isGenerating = true;
    try {
        // Add assistant message with loading indicator
        const responseElement = addMessage('', true, 'assistant');
        
        // First, save the user message to the database
        if (currentConversationId) {
            try {
                const userMsgResponse = await fetch(`/api/conversations/${currentConversationId}/messages`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        role: 'user',
                        content: message,
                        attachment_filename: sentAttachmentFilename
                    })
                });
                
                const userMsgData = await userMsgResponse.json();
                if (userMsgData.status !== 'success') {
                    console.error('Failed to save user message:', userMsgData.message);
                }
            } catch (msgError) {
                console.error('Error saving user message:', msgError);
            }
        }
        
        // Use fetch with streaming
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                ...modelConfig,
                conversation_id: currentConversationId,
                attachment_filename: sentAttachmentFilename // Send attached filename
            })
        });
        
        // Check if response is not ok (error)
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error generating response:', errorData);
            
            // Check if the error is due to quota exceeded
            if (errorData.quota_exceeded) {
                // Don't remove the user message, keep it in the chat
                
                // Create a quota exceeded message for the chat
                const quotaInfo = errorData.quota_info || {};
                let quotaMessage = 'Message quota exceeded. ';
                
                // Add specific reason if available
                if (quotaInfo.reason === 'daily_limit') {
                    quotaMessage += `You've reached your daily message limit (${quotaInfo.used || 0}/${quotaInfo.limit || 0}).`;
                } else if (quotaInfo.reason === 'monthly_limit') {
                    quotaMessage += `You've reached your monthly message limit (${quotaInfo.used || 0}/${quotaInfo.limit || 0}).`;
                }
                
                // Add the message to the chat
                addMessage(quotaMessage, false, 'system');
                
                // Show quota exceeded modal
                showQuotaExceededModal(quotaInfo);
                
                isGenerating = false;
                updateSendButtonState();
                return;
            } else {
                // Handle other errors
                addMessage(`Error: ${errorData.message || 'Failed to generate response'}`, false, 'system');
                isGenerating = false;
                updateSendButtonState();
                return;
            }
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let fullResponse = '';
        let thinkingContent = '';
        let finalUserMessageId = null; // To store the ID of the saved user message
        let finalAssistantMessageId = null; // To store the ID of the saved assistant message
        let titleUpdateProcessed = false; // Flag to prevent duplicate title updates
        
        // Remove loading indicator and prepare containers
        responseElement.innerHTML = '';
        
        // Create containers for thinking and main response
        const thinkingElement = document.createElement('details');
        thinkingElement.className = 'thinking';
        thinkingElement.open = userPreferences.show_thinking || false;
        
        const summary = document.createElement('summary');
        summary.textContent = 'Thinking Process';
        thinkingElement.appendChild(summary);
        
        const thinkingContentEl = document.createElement('div');
        thinkingContentEl.className = 'thinking-content';
        thinkingElement.appendChild(thinkingContentEl);
        
        const mainResponse = document.createElement('div');
        mainResponse.className = 'main-response';
        
        // Add containers to the message element
        responseElement.appendChild(thinkingElement);
        responseElement.appendChild(mainResponse);
        
        // Initially hide the thinking element until we have content
        thinkingElement.style.display = 'none';
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        
                        if (data.error) {
                            // responseElement.remove();
                            addMessage(`Error: ${data.error}`, false, 'system');
                            break;
                        }
                        
                        if (!data.done) {
                            // Update the response as it streams in
                            fullResponse += data.chunk;
                            
                            // Extract thinking content if present
                            if (fullResponse.includes('<think>') && fullResponse.includes('</think>')) {
                                const thinkStart = fullResponse.indexOf('<think>') + '<think>'.length;
                                const thinkEnd = fullResponse.indexOf('</think>');
                                
                                if (thinkStart > 0 && thinkEnd > thinkStart) {
                                    thinkingContent = fullResponse.substring(thinkStart, thinkEnd).trim();
                                    thinkingContentEl.innerHTML = renderMarkdown(thinkingContent);
                                    
                                    // Show thinking element if it has content
                                    if (thinkingElement.style.display === 'none') {
                                        thinkingElement.style.display = 'block';
                                    }
                                }
                            }
                            
                            // Update main response (excluding thinking content)
                            let displayResponse = fullResponse;
                            if (displayResponse.includes('<think>') && displayResponse.includes('</think>')) {
                                displayResponse = displayResponse.replace(/<think>[\s\S]*?<\/think>/g, '');
                            }
                            
                            // Remove common prefixes
                            const prefixes = ['Answer:', 'Final Answer:', 'My answer:', "Here's my answer:", 'The answer is:'];
                            for (const prefix of prefixes) {
                                if (displayResponse.trim().startsWith(prefix)) {
                                    displayResponse = displayResponse.trim().substring(prefix.length).trim();
                                }
                            }
                            
                            // Update the main response content
                            mainResponse.innerHTML = formatMarkdown(displayResponse);
                            
                            // Scroll to bottom as content is added
                            scrollToBottom();
                        } else if (data.title_update && !titleUpdateProcessed) {
                            // Track that we're processing this update
                            titleUpdateProcessed = true;
                            
                            // Handle title update event from server
                            console.log('Received title update from server:', data.title);
                            
                            // Update the UI with the new title
                            updatePageTitle(data.title);
                            updateHeaderInfo(data.title, currentModel);
                            
                            // Update conversation list to show the new title
                            loadConversations(true); // Force refresh
                            
                            // Reset flag after slight delay to allow UI updates
                            setTimeout(() => { titleUpdateProcessed = false; }, 1000);
                        } else {
                            // Final message with complete response
                            // Set final content
                            const finalContent = data.full_response || fullResponse;
                            finalUserMessageId = data.user_message_id; // Get user message ID
                            finalAssistantMessageId = data.assistant_message_id; // Get assistant message ID
                            
                            // Update final thinking content
                            const finalThinking = data.thinking || thinkingContent;
                            if (finalThinking && finalThinking.trim() !== '') {
                                // Update thinking content
                                thinkingContentEl.innerHTML = renderMarkdown(finalThinking);
                                
                                // Show thinking element if it has content
                                thinkingElement.style.display = 'block';
                            } else {
                                // Hide thinking element if no content
                                thinkingElement.style.display = 'none';
                            }
                            
                            // Update final response content
                            mainResponse.innerHTML = renderMarkdown(finalContent);
                            
                            // Add the DB ID to the assistant element and history
                            if (finalAssistantMessageId) {
                                responseElement.dataset.messageId = finalAssistantMessageId;
                                // Find the corresponding history item using the DOM ID's timestamp part
                                const assistantHistoryItem = messageHistory.find(m => responseElement.id.endsWith(m.id));
                                if (assistantHistoryItem) {
                                    assistantHistoryItem.dbId = finalAssistantMessageId;
                                } else {
                                     console.warn("Could not find assistant message in history to update dbId:", responseElement.id);
                                }
                            }
                            
                            // Add the DB ID to the user element and history
                            if (finalUserMessageId && userMessageElement) {
                                userMessageElement.dataset.messageId = finalUserMessageId;
                                // Find the corresponding history item using the DOM ID's timestamp part
                                const userHistoryItem = messageHistory.find(m => userMessageElement.id.endsWith(m.id));
                                if (userHistoryItem) {
                                    userHistoryItem.dbId = finalUserMessageId;
                                } else {
                                     console.warn("Could not find user message in history to update dbId:", userMessageElement.id);
                                }
                            }

                            // Ensure buttons are added after final content render for assistant
                            addMessageButtons(responseElement, 'assistant');
                            
                            // Find message in history and update with final content and thinking
                            // (Already updated dbId above)
                            const messageHistoryItem = messageHistory.find(m => responseElement.id.endsWith(m.id));
                            if (messageHistoryItem) {
                                messageHistoryItem.content = finalContent;
                                messageHistoryItem.thinking = finalThinking;
                            }
                            
                            // Update conversation list to show latest message
                            loadConversations(true); // Force refresh
                        }
                    } catch (e) {
                        console.error('Error parsing streaming data:', e, line);
                    }
                }
            }
        }
    } catch (error) {
        addMessage(`Error: ${error.message}`, false, 'system');
        console.error('Error generating response:', error);
    } finally {
        isGenerating = false;
        scrollToBottom();
        removeAttachment(); // Clear attachment preview after sending
        
        // Generate title for new conversations
        if (isNewConversation && currentConversationId) {
            console.log("Generating title for new conversation:", currentConversationId);
            try {
                // First update conversation list to show latest message
                loadConversations(true);
                
                // Use the same model that was used for the response
                // If it's a special model like a bot or workflow, use the default text model
                let modelForTitleGeneration = selectedModel;
                
                // Check if the selected model is valid for title generation
                const selectedModelInfo = availableModels.find(m => m.id === selectedModel);
                
                // If the model doesn't exist or doesn't support text generation, find a default text model
                if (!selectedModelInfo || (selectedModelInfo.capabilities && !selectedModelInfo.capabilities.includes('text-generation'))) {
                    // Find the first available text model
                    const textModel = availableModels.find(m => m.capabilities && m.capabilities.includes('text-generation'));
                    if (textModel) {
                        modelForTitleGeneration = textModel.id;
                        console.log("Using alternative text model", modelForTitleGeneration, "for title generation");
                    } else {
                        // If no text model is available, use the selected model anyway
                        console.log("No text model found, using selected model", selectedModel, "for title generation");
                    }
                } else {
                    console.log("Using current model", modelForTitleGeneration, "for title generation");
                }
                
                // Generate a title based on the conversation
                const titleResponse = await fetch(`/api/conversations/${currentConversationId}/generate-title?model=${modelForTitleGeneration}`, {
                    method: 'POST'
                });
                
                if (titleResponse.ok) {
                    const titleData = await titleResponse.json();
                    
                    if (titleData.status === 'success' && titleData.title) {
                        console.log("Title generated successfully:", titleData.title);
                        // Update the title in the UI
                        updatePageTitle(titleData.title);
                        updateHeaderInfo(titleData.title, selectedModel);
                        
                        // Reload conversations to show new title
                        loadConversations(true);
                    } else {
                        console.warn("Title generation returned an error:", titleData.message || "Unknown error");
                        // Still reload conversations to show updated list
                        loadConversations(true);
                    }
                } else {
                    console.warn("Title generation request failed with status:", titleResponse.status);
                    // Still reload conversations to show updated list
                    loadConversations(true);
                }
            } catch (titleError) {
                console.error('Error generating title:', titleError);
                // Still reload conversations even if title generation fails
                loadConversations(true);
            } finally {
                // Reset the new conversation flag
                isNewConversation = false;
            }
        }
    }
}

// Update the page title with the conversation title
function updatePageTitle(title) {
    document.title = title ? `${title} - MIDAS3.0` : 'MIDAS3.0';
    
    // If there's a header title element, update it too
    if (chatTitleElement) {
        chatTitleElement.textContent = title || 'New Chat';
    }
}

// Update header info with current chat title and model
function updateHeaderInfo(title, model) {
    // Update title
    if (chatTitleElement) {
        chatTitleElement.textContent = title || 'New Chat';
    }
    
    // Update model
    if (chatModelElement && model) {
        // Check if this is a bot model
        if (model.startsWith('bot:')) {
            // Extract bot ID
            const botId = model.substring(4);
            
            // Fetch the bot details to get its name
            fetch(`/api/bots/${botId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // Display the bot name instead of the ID without the "Bot:" prefix
                        chatModelElement.textContent = data.bot.name;
                    } else {
                        // Fallback if bot details can't be fetched
                        chatModelElement.textContent = model.replace('bot:', '');
                    }
                })
                .catch(error => {
                    console.error('Error fetching bot details:', error);
                    chatModelElement.textContent = model.replace('bot:', '');
                });
        } else {
            // Regular model, display as is (remove any workflow: prefix)
            chatModelElement.textContent = model.replace('workflow:', '');
        }
    } else if (chatModelElement) {
        chatModelElement.textContent = 'Select a model';
    }
}

// Update conversation in list
function updateConversationInList(conversationId) {
    const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
    if (conversationItem) {
        conversationItem.querySelector('.conversation-title').textContent = 'New Title';
    }
}

// Update conversation in list with animation
function updateConversationInList(conversationId) {
    const item = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
    if (item) {
        item.classList.add('updated');
        
        // Remove the class after animation completes
        setTimeout(() => {
            item.classList.remove('updated');
        }, 500);
    }
}

// Call this function whenever a conversation is updated
// For example, after title generation or message sending
loadConversations = async function(refresh = false) {
    try {
        showLoading('Loading conversations...');
        
        const response = await fetch('/api/conversations');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Sort by updated_at descending
            const sortedConversations = data.conversations.sort((a, b) => 
                new Date(b.updated_at) - new Date(a.updated_at)
            );
            
            // Get most recent 5 conversations
            const recentConversations = sortedConversations.slice(0, 5);
            
            // Clear existing conversations
            conversationList.innerHTML = '';
            
            // Add recent conversations
            recentConversations.forEach(conversation => {
                const item = document.createElement('div');
                item.className = 'conversation-item';
                item.dataset.id = conversation.id;
                
                if (conversation.id == currentConversationId) {
                    item.classList.add('active-conversation');
                }
                
                // Create title element
                const titleEl = document.createElement('div');
                titleEl.className = 'conversation-title';
                titleEl.textContent = conversation.title || 'New Chat';
                
                // Create metadata container
                const metaContainer = document.createElement('div');
                metaContainer.className = 'conversation-meta';
                
                // Create model element
                const modelEl = document.createElement('span');
                modelEl.className = 'conversation-model';
                
                // Check if this is a bot model
                if (conversation.model && conversation.model.startsWith('bot:')) {
                    // Extract bot ID
                    const botId = conversation.model.substring(4);
                    
                    // Set a temporary value
                    modelEl.textContent = 'Bot';
                    
                    // Fetch the bot details to get its name
                    fetch(`/api/bots/${botId}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                // Display the bot name instead of the ID
                                modelEl.textContent = data.bot.name;
                            } else {
                                // Fallback if bot details can't be fetched
                                modelEl.textContent = conversation.model;
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching bot details:', error);
                            modelEl.textContent = conversation.model;
                        });
                } else {
                    // Regular model, display as is (remove any workflow: prefix)
                    modelEl.textContent = conversation.model.replace('workflow:', '');
                }
                
                // Create date element
                const dateEl = document.createElement('span');
                dateEl.className = 'conversation-date';
                dateEl.textContent = new Date(conversation.updated_at).toLocaleString();
                
                // Add elements to container
                metaContainer.appendChild(modelEl);
                metaContainer.appendChild(dateEl);
                
                item.appendChild(titleEl);
                item.appendChild(metaContainer);
                
                // Add buttons container
                const buttonsContainer = document.createElement('div');
                buttonsContainer.className = 'conversation-buttons';
                
                // Add rename button
                const renameButton = document.createElement('button');
                renameButton.className = 'rename-conversation-button';
                renameButton.innerHTML = '<i class="fas fa-edit"></i>';
                renameButton.title = 'Rename';
                renameButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    renameConversation(conversation.id, conversation.title);
                });
                
                // Add delete button
                const deleteButton = document.createElement('button');
                deleteButton.className = 'delete-conversation-button';
                deleteButton.innerHTML = '<i class="fas fa-trash"></i>';
                deleteButton.title = 'Delete';
                deleteButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    currentConversationToDelete = conversation.id;
                    document.getElementById('delete-confirm-modal').style.display = 'flex';
                });
                
                buttonsContainer.appendChild(renameButton);
                buttonsContainer.appendChild(deleteButton);
                item.appendChild(buttonsContainer);
                
                // Add click handler
                item.addEventListener('click', (e) => {
                    // Only load conversation if clicking on the item itself, not buttons
                    if (e.target === item || e.target.classList.contains('conversation-title') || 
                        e.target.classList.contains('conversation-meta')) {
                        loadConversation(conversation.id);
                    }
                });
                
                conversationList.appendChild(item);
            });
            
            // Add 'More' button if there are more conversations
            if (sortedConversations.length > 5) {
                const moreButton = document.createElement('div');
                moreButton.className = 'more-conversations-button';
                moreButton.textContent = '... More Chats';
                moreButton.addEventListener('click', showAllConversations);
                conversationList.appendChild(moreButton);
            }
            
            // Update conversation in list with animation
            updateConversationInList(currentConversationId);
        } else {
            addMessage(`Error: ${data.message}`, false, 'system');
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
        conversationList.innerHTML = '<div class="empty-message">Failed to load conversations</div>';
    } finally {
        removeLoading();
    }
}

// --- Document Upload Handler ---
async function handleDocumentUpload() {
    if (!docUploadInput || !docUploadInput.files || docUploadInput.files.length === 0) {
        uploadStatus.textContent = 'Please select a file first.';
        setTimeout(() => uploadStatus.textContent = '', 3000);
        return;
    }

    // Check if there's a current conversation
    if (!currentConversationId) {
        uploadStatus.textContent = 'Please start a conversation first.';
        showNotification('Please start a conversation before uploading documents', 'error');
        setTimeout(() => uploadStatus.textContent = '', 3000);
        return;
    }

    const file = docUploadInput.files[0];
    const allowedTypes = ['text/plain', 'application/pdf', 'text/markdown', 'text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    const allowedExtensions = ['.txt', '.pdf', '.md', '.csv', '.xls', '.xlsx'];
    const fileExtension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();

    // Basic client-side check for extension
    if (!allowedExtensions.includes(fileExtension)) {
        uploadStatus.textContent = 'Invalid file type. Allowed: ' + allowedExtensions.join(', ');
        setTimeout(() => uploadStatus.textContent = '', 4000);
        docUploadInput.value = ''; // Clear the hidden file input
        return;
    }

    const formData = new FormData();
    
    // Add the file with the correct field name 'document' that the backend expects
    formData.append('document', file);
    
    // Add the conversation ID to the form data
    formData.append('conversation_id', currentConversationId);

    console.log('FormData keys before fetch:');
    for (let key of formData.keys()) {
        console.log(key);
    }
    // Debugging: Log the file object itself
    console.log('File object being sent:', file);
    console.log('Uploading to conversation:', currentConversationId);

    uploadStatus.textContent = `Uploading ${file.name}...`;
    fileUploadButton.disabled = true;
    
    // Add a visual indicator to the file upload button
    fileUploadButton.classList.add('uploading');
    fileUploadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    try {
        const response = await fetch('/api/upload-doc', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            console.log('Document uploaded successfully:', result);
            // Show attachment preview
            currentAttachmentFilename = file.name;
            displayAttachmentPreview(currentAttachmentFilename);
            uploadStatus.textContent = `${file.name} attached`;
            uploadStatus.style.display = 'block';
            showNotification(`${file.name} added to this conversation's knowledge base!`, 'success');
            addMessage(`Document "${file.name}" has been uploaded and added to this conversation's knowledge base.`, false, 'system');
            setTimeout(() => {
                uploadStatus.style.display = 'none';
            }, 3000);
        } else {
            uploadStatus.textContent = `Error: ${result.message || 'Upload failed'}`;
            showNotification(`Error: ${result.message || 'Upload failed'}`, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = 'Upload failed. See console for details.';
        showNotification('Upload failed. See console for details.', 'error');
    } finally {
        // Reset the file upload button
        fileUploadButton.disabled = false;
        fileUploadButton.classList.remove('uploading');
        fileUploadButton.innerHTML = '<i class="fas fa-paperclip"></i>';
        
        // Clear status message after a delay
        setTimeout(() => uploadStatus.textContent = '', 5000);
    }
}

// --- Attachment Preview --- 
function displayAttachmentPreview(filename) {
    if (filename) {
        attachmentPreviewFilename.textContent = filename;
        attachmentPreviewFilename.title = filename; // Show full name on hover
        attachmentPreview.style.display = 'flex';
        messageInput.style.paddingRight = '50px'; // Add padding to avoid text overlapping with attachment
        updateSendButtonState(); // Enable send button if there's an attachment
    } else {
        attachmentPreview.style.display = 'none';
        messageInput.style.paddingRight = ''; // Reset padding
    }
}

function removeAttachment() {
    currentAttachmentFilename = null;
    docUploadInput.value = ''; // Clear the hidden file input
    attachmentPreview.style.display = 'none';
    attachmentPreviewFilename.textContent = '';
    attachmentPreviewFilename.title = '';
    messageInput.style.paddingRight = ''; // Reset padding
    updateSendButtonState(); // Re-evaluate send button state
}

// --- Helper Functions ---
function updateSendButtonState() {
    const message = messageInput.value.trim();
    
    // Enable if message has content OR an attachment is present
    sendButton.disabled = !(message || currentAttachmentFilename);
}

// Function to initialize preferences UI and listeners
function initializePreferences() {
    const preferencesButton = document.getElementById('preferences-button');
    const preferencesModal = document.getElementById('preferences-modal');
    const savePreferencesButton = document.getElementById('save-preferences');
    const cancelPreferencesButton = document.getElementById('cancel-preferences');
    const tabButtons = document.querySelectorAll('.tab-button');
    const preferencesLoading = document.getElementById('preferences-loading');
    const modelsTab = document.getElementById('models-tab');
    
    // Show preferences modal when preferences button is clicked
    preferencesButton.addEventListener('click', function() {
        // Show the modal first
        preferencesModal.style.display = 'flex';
        
        // Show loading animation, hide tabs content
        preferencesLoading.style.display = 'flex';
        document.querySelectorAll('.tab-content').forEach(content => {
            content.style.display = 'none';
        });
        
        // Reset tab buttons to ensure Models tab is active
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabButtons[0].classList.add('active'); // First tab button is Models
        
        // Load preferences (will hide loading when done)
        loadPreferences();
    });
    
    // Hide modal when cancel button is clicked
    cancelPreferencesButton.addEventListener('click', function() {
        preferencesModal.style.display = 'none';
    });
    
    // Save preferences when save button is clicked
    savePreferencesButton.addEventListener('click', function() {
        savePreferences();
    });
    
    // Close modal when clicking outside
    preferencesModal.addEventListener('click', function(event) {
        if (event.target === preferencesModal && confirm("Are you sure you want to close without saving?")) {
            preferencesModal.style.display = 'none';
        }
    });
    
    // Tab switching functionality
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all tabs
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Get the tab to show
            const tabToShow = this.getAttribute('data-tab');
            
            // Hide all tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.style.display = 'none';
            });
            
            // Show the selected tab content
            document.getElementById(`${tabToShow}-tab`).style.display = 'block';
        });
    });
    
    // Initialize theme from localStorage or apply default
    initializeTheme();
}

// Function to save user preferences
async function savePreferences() {
    const defaultModel = document.getElementById('default-model').value;
    const defaultEmbeddingModel = document.getElementById('default-embedding-model').value;
    const visibleModelsCheckboxes = document.querySelectorAll('#visible-models-list input[type="checkbox"]');
    const themeSelect = document.getElementById('theme-select');
    const showThinking = document.getElementById('show-thinking').checked;

    // Get visible models from checkboxes
    const visibleModels = Array.from(visibleModelsCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    // Get theme from select
    const theme = themeSelect ? themeSelect.value : 'system';

    // Prepare the preferences object
    const updatedPreferences = {
        default_model: defaultModel || null,
        default_embedding_model: defaultEmbeddingModel,
        visible_models: visibleModels,
        theme: theme,
        show_thinking: showThinking
    };

    try {
        const response = await fetch('/api/preferences', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updatedPreferences)
        });

        const data = await response.json();
        if (data.status === 'success') {
            // Update local preferences
            userPreferences = updatedPreferences;
            // Close the modal
            document.getElementById('preferences-modal').style.display = 'none';
            // Apply theme
            applyTheme(theme);
            // Show success message
            showNotification('Preferences saved successfully', 'success');
            
            // Reload preferences to update UI
            await loadPreferences();
            
            // Reload models to update dropdown
            await loadModels();
            
            // Apply new default model if set
            if (updatedPreferences.default_model && modelSelect) {
                modelSelect.value = updatedPreferences.default_model;
                currentModel = updatedPreferences.default_model;
            }
        } else {
            throw new Error(data.message || 'Failed to save preferences');
        }
    } catch (error) {
        console.error('Error saving preferences:', error);
        showNotification('Failed to save preferences: ' + error.message, 'error');
    }
}

// Function to load preferences from the server
async function loadPreferences() {
    try {
        const preferencesLoading = document.getElementById('preferences-loading');
        const modelsTab = document.getElementById('models-tab');
        
        // Show loading animation
        preferencesLoading.style.display = 'flex';
        
        const response = await fetch('/api/preferences');
        const data = await response.json();
        
        if (data.status === 'success') {
            userPreferences = data.preferences;
            availableModels = data.available_models;
            availableEmbeddingModels = data.embedding_models;
            
            // Populate UI with preferences
            populatePreferencesUI();
            
            // Hide loading, show models tab (always start with models tab)
            preferencesLoading.style.display = 'none';
            modelsTab.style.display = 'block';
            
            // Make sure appearance tab is hidden
            document.getElementById('appearance-tab').style.display = 'none';
        } else {
            console.error('Error loading preferences:', data.message);
            
            // Hide loading, show error message
            preferencesLoading.innerHTML = `
                <p class="error-message">Error loading preferences. Please try again.</p>
                <button class="retry-button" onclick="loadPreferences()">Retry</button>
            `;
        }
    } catch (error) {
        console.error('Error fetching preferences:', error);
        
        // Hide loading, show error message
        const preferencesLoading = document.getElementById('preferences-loading');
        preferencesLoading.innerHTML = `
            <div class="error-state">
                <p>Error loading preferences. Please try again.</p>
                <button class="retry-button" onclick="loadPreferences()">Retry</button>
            </div>
        `;
    }
}

// Function to populate the preferences UI with current values
function populatePreferencesUI() {
    // Populate default model dropdown
    const defaultModelSelect = document.getElementById('default-model');
    defaultModelSelect.innerHTML = '';
    
    // Add "None" option
    const noneOption = document.createElement('option');
    noneOption.value = '';
    noneOption.textContent = 'None (use first available)';
    defaultModelSelect.appendChild(noneOption);
    
    // Sort models to put the default model first
    const sortedModels = [...availableModels];
    if (userPreferences.default_model) {
        // Remove the default model from the array
        const defaultModelIndex = sortedModels.findIndex(model => model === userPreferences.default_model);
        if (defaultModelIndex !== -1) {
            const defaultModel = sortedModels.splice(defaultModelIndex, 1)[0];
            // Add it back at the beginning
            sortedModels.unshift(defaultModel);
        }
    }
    
    // Add all available models
    sortedModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        
        // Set selected if it matches the current default
        if (userPreferences.default_model === model) {
            option.selected = true;
        }
        
        defaultModelSelect.appendChild(option);
    });
    
    // Populate embedding model dropdown
    const embeddingModelSelect = document.getElementById('default-embedding-model');
    embeddingModelSelect.innerHTML = '';
    
    availableEmbeddingModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        
        // Set selected if it matches the current default
        if (userPreferences.default_embedding_model === model) {
            option.selected = true;
        }
        
        embeddingModelSelect.appendChild(option);
    });
    
    // Populate visible models checkboxes
    const visibleModelsContainer = document.getElementById('visible-models-container');
    visibleModelsContainer.innerHTML = '';
    
    availableModels.forEach(model => {
        const label = document.createElement('label');
        label.className = 'checkbox-label';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = model;
        checkbox.checked = userPreferences.visible_models.length === 0 || userPreferences.visible_models.includes(model);
        
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(' ' + model));
        
        visibleModelsContainer.appendChild(label);
    });
    
    // Set theme radio buttons
    const themeRadios = document.querySelectorAll('input[name="theme"]');
    themeRadios.forEach(radio => {
        if (radio.value === userPreferences.theme) {
            radio.checked = true;
        }
    });
    
    // Set thinking process checkbox
    const showThinkingCheckbox = document.getElementById('show-thinking');
    showThinkingCheckbox.checked = userPreferences.show_thinking || false;
}

// Function to save preferences to the server
// Make it globally accessible so it can be called from auth.js
window.savePreferences = async function() {
    try {
        // Get values from UI
        const defaultModel = document.getElementById('default-model').value;
        const defaultEmbeddingModel = document.getElementById('default-embedding-model').value;
        
        // Get visible models from checkboxes
        const visibleModels = [];
        const visibleModelCheckboxes = document.querySelectorAll('#visible-models-container input[type="checkbox"]:checked');
        visibleModelCheckboxes.forEach(checkbox => {
            visibleModels.push(checkbox.value);
        });
        
        // Get theme from radio buttons
        const themeRadio = document.querySelector('input[name="theme"]:checked');
        const theme = themeRadio ? themeRadio.value : 'light';
        
        // Get thinking process preference
        const showThinking = document.getElementById('show-thinking').checked;
        
        // Update preferences object
        const updatedPreferences = {
            default_model: defaultModel || null,
            default_embedding_model: defaultEmbeddingModel,
            visible_models: visibleModels,
            theme: theme,
            show_thinking: showThinking
        };
        
        // Send to server
        const response = await fetch('/api/preferences', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updatedPreferences)
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update local preferences
            userPreferences = updatedPreferences;
            
            // Apply theme
            applyTheme(theme);
            
            // Close modal
            document.getElementById('preferences-modal').style.display = 'none';
            
            // Show success notification
            showNotification('Preferences saved successfully');
            
            // Reload models in the dropdown
            loadModels();
        } else {
            console.error('Error saving preferences:', data.message);
            showNotification('Error saving preferences', 'error');
        }
    } catch (error) {
        console.error('Error saving preferences:', error);
        showNotification('Error saving preferences', 'error');
    }
}

// Helper function to get selected visible models
function getSelectedVisibleModels() {
    const selectedModels = [];
    const checkboxes = document.querySelectorAll('#visible-models-container input[type="checkbox"]:checked');
    
    // If all models are selected, return an empty array (show all)
    if (checkboxes.length === availableModels.length) {
        return [];
    }
    
    // Otherwise, return the selected models
    checkboxes.forEach(checkbox => {
        selectedModels.push(checkbox.value);
    });
    
    return selectedModels;
}

// Helper function to get selected theme
function getSelectedTheme() {
    const selectedRadio = document.querySelector('input[name="theme"]:checked');
    return selectedRadio ? selectedRadio.value : 'light';
}

// Function to apply theme
function applyTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    } else if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    } else if (theme === 'system') {
        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
        }
        localStorage.setItem('theme', 'system');
    }
}

// Function to show notification
function showNotification(message, type = 'success') {
    // Create notification element if it doesn't exist
    let notification = document.getElementById('notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        document.body.appendChild(notification);
    }
    
    // Set content and type
    notification.textContent = message;
    notification.className = `notification ${type}`;
    
    // Show notification
    notification.style.display = 'block';
    
    // Hide after 3 seconds
    setTimeout(() => {
        notification.style.display = 'none';
    }, 3000);
}

// Function to initialize theme
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);
}

// Populate model dropdowns for bot editor
async function populateModelDropdowns() {
    try {
        // Fetch base models
        const modelsResponse = await fetch('/api/models');
        const modelsData = await modelsResponse.json();
        
        // Fetch embedding models
        const embeddingResponse = await fetch('/api/embedding_models');
        const embeddingData = await embeddingResponse.json();
        
        // Get the select elements
        const baseModelSelect = document.getElementById('bot-base-model');
        const embeddingModelSelect = document.getElementById('bot-embedding-model');
        
        // Clear existing options except the first one (placeholder)
        while (baseModelSelect.options.length > 1) {
            baseModelSelect.remove(1);
        }
        
        while (embeddingModelSelect.options.length > 1) {
            embeddingModelSelect.remove(1);
        }
        
        // Add base models
        if (modelsData.status === 'success' && modelsData.models) {
            modelsData.models.forEach(model => {
                const option = document.createElement('option');
                // Handle both string and object formats
                const modelValue = typeof model === 'object' ? model.name || model.id : model;
                const modelText = typeof model === 'object' ? model.name || model.id : model;
                
                option.value = modelValue;
                option.textContent = modelText;
                baseModelSelect.appendChild(option);
            });
        }
        
        // Add embedding models
        if (embeddingData.status === 'success' && embeddingData.models) {
            embeddingData.models.forEach(model => {
                const option = document.createElement('option');
                // Handle both string and object formats
                const modelValue = typeof model === 'object' ? model.name || model.id : model;
                const modelText = typeof model === 'object' ? model.name || model.id : model;
                
                option.value = modelValue;
                option.textContent = modelText;
                embeddingModelSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error fetching models for bot editor:', error);
        // Show error message
        showNotification('Failed to load models. Please try again.', 'error');
    }
}

// ============================================================
// Bot Management
// ============================================================

// Initialize bot management UI
function initializeBotManagement() {
    const manageBotButton = document.getElementById('manage-bots-button');
    const botManagementModal = document.getElementById('bot-management-modal');
    const closeBotModalButton = document.getElementById('close-bot-modal');
    const createNewBotButton = document.getElementById('create-new-bot-button');
    const backToListButton = document.getElementById('back-to-bot-list-button');
    const botListSection = document.getElementById('bot-list-section');
    const botEditorSection = document.getElementById('bot-editor-section');
    const botForm = document.getElementById('bot-form');
    const cancelBotEditButton = document.getElementById('cancel-bot-edit');
    const uploadKnowledgeButton = document.getElementById('upload-knowledge-button');
    const knowledgeFileInput = document.getElementById('knowledge-file-input');
    
    // Temperature and Top-P sliders
    const temperatureSlider = document.getElementById('bot-temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const topPSlider = document.getElementById('bot-top-p');
    const topPValue = document.getElementById('top-p-value');
    
    // Update temperature value display when slider changes
    temperatureSlider.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
    });
    
    // Update top-p value display when slider changes
    topPSlider.addEventListener('input', function() {
        topPValue.textContent = this.value;
    });
    
    // Open bot management modal
    manageBotButton.addEventListener('click', function() {
        botManagementModal.style.display = 'flex';
        loadBots();
    });
    
    // Close bot management modal
    closeBotModalButton.addEventListener('click', function() {
        botManagementModal.style.display = 'none';
    });
    
    // Close modal when clicking outside
    botManagementModal.addEventListener('click', function(event) {
        if (event.target === botManagementModal && confirm("Are you sure you want to close the bot management panel?")) {
            botManagementModal.style.display = 'none';
        }
    });
    
    // Show bot editor for creating a new bot
    createNewBotButton.addEventListener('click', function() {
        showBotEditor();
    });
    
    // Go back to bot list
    backToListButton.addEventListener('click', function() {
        showBotList();
    });
    
    // Cancel bot edit
    cancelBotEditButton.addEventListener('click', function() {
        showBotList();
    });
    
    // Handle bot form submission
    botForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        event.stopPropagation();
        await saveBot();
        return false;
    });
    
    // Handle knowledge file upload button
    uploadKnowledgeButton.addEventListener('click', function() {
        knowledgeFileInput.click();
    });
    
    // Handle re-index knowledge button
    const reindexKnowledgeButton = document.getElementById('reindex-knowledge-button');
    if (reindexKnowledgeButton) {
        reindexKnowledgeButton.addEventListener('click', function() {
            const botId = document.getElementById('bot-id').value;
            if (botId) {
                reindexKnowledgeFiles(botId);
            }
        });
    }
    
    // Handle knowledge file selection
    knowledgeFileInput.addEventListener('change', function() {
        const botId = document.getElementById('bot-id').value;
        if (botId && this.files.length > 0) {
            uploadKnowledgeFiles(botId, this.files);
        }
    });
}

// Load bots from the server
async function loadBots() {
    const botListContainer = document.getElementById('bot-list-container');
    
    try {
        // Show loading
        botListContainer.innerHTML = `
            <div class="loading-container">
                <img src="/static/assets/xeno.png" alt="Loading" class="loading-spinner">
                <p>Loading bots...</p>
            </div>
        `;
        
        const response = await fetch('/api/bots');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Clear loading
            botListContainer.innerHTML = '';
            
            if (data.bots.length === 0) {
                botListContainer.innerHTML = `
                    <div class="empty-state">
                        <img src="/static/assets/monoXenosad.png" alt="No Bots" class="empty-state-image" style="width: 80px; height: auto;">
                        <p>You don't have any bots yet.</p>
                        <p>Create a new bot to get started!</p>
                    </div>
                `;
                return;
            }
            
            // Add bots to the list
            data.bots.forEach(bot => {
                const botItem = createBotListItem(bot);
                botListContainer.appendChild(botItem);
            });
        } else {
            botListContainer.innerHTML = `
                <div class="error-state">
                    <p>Error loading bots: ${data.message}</p>
                    <button class="retry-button" onclick="loadBots()">Retry</button>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading bots:', error);
        
        // Hide loading, show error message
        const botListContainer = document.getElementById('bot-list-container');
        botListContainer.innerHTML = `
            <div class="error-state">
                <p>Error loading bots. Please try again.</p>
                <button class="retry-button" onclick="loadBots()">Retry</button>
            </div>
        `;
    }
}

// Create a bot list item
function createBotListItem(bot) {
    const botItem = document.createElement('div');
    botItem.className = 'bot-item';
    botItem.dataset.id = bot.id;
    
    const botInfo = document.createElement('div');
    botInfo.className = 'bot-info';
    
    const botName = document.createElement('div');
    botName.className = 'bot-name';
    botName.textContent = bot.name;
    
    const botDescription = document.createElement('div');
    botDescription.className = 'bot-description';
    botDescription.textContent = bot.description || 'No description';
    
    const botModel = document.createElement('div');
    botModel.className = 'bot-model';
    botModel.textContent = `Model: ${bot.base_model || 'Not set'}`;
    
    botInfo.appendChild(botName);
    botInfo.appendChild(botDescription);
    botInfo.appendChild(botModel);
    
    const botActions = document.createElement('div');
    botActions.className = 'bot-actions';
    
    const editButton = document.createElement('button');
    editButton.innerHTML = '<i class="fas fa-edit"></i>';
    editButton.title = 'Edit Bot';
    editButton.addEventListener('click', (e) => {
        e.stopPropagation();
        editBot(bot.id);
    });
    
    const deleteButton = document.createElement('button');
    deleteButton.innerHTML = '<i class="fas fa-trash"></i>';
    deleteButton.title = 'Delete Bot';
    deleteButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        confirmDeleteBot(bot.id);
    });
    
    botActions.appendChild(editButton);
    botActions.appendChild(deleteButton);
    botItem.appendChild(botInfo);
    botItem.appendChild(botActions);
    
    // Add click handler to open chat with this bot
    botItem.addEventListener('click', function() {
        startChatWithBot(bot.id);
    });
    
    return botItem;
}

// Show bot editor
async function showBotEditor(botId = null) {
    const botListSection = document.getElementById('bot-list-section');
    const botEditorSection = document.getElementById('bot-editor-section');
    const botModalTitle = document.getElementById('bot-modal-title');
    const createNewBotButton = document.getElementById('create-new-bot-button');
    const backToListButton = document.getElementById('back-to-bot-list-button');
    const botForm = document.getElementById('bot-form');
    const botBaseModelSelect = document.getElementById('bot-base-model');
    const botEmbeddingModelSelect = document.getElementById('bot-embedding-model');
    
    // Show editor section, hide list section
    botListSection.style.display = 'none';
    botEditorSection.style.display = 'block';
    
    // Update header
    createNewBotButton.style.display = 'none';
    backToListButton.style.display = 'inline-block';
    
    // Update title
    if (botId) {
        botModalTitle.textContent = 'Edit Bot';
    } else {
        botModalTitle.textContent = 'Create New Bot';
    }
    
    // Reset form
    botForm.reset();
    document.getElementById('bot-id').value = '';
    document.getElementById('knowledge-files-list').innerHTML = '';
    
    // Populate model dropdowns
    populateModelDropdowns();
    
    if (botId) {
        // Editing existing bot
        try {
            const response = await fetch(`/api/bots/${botId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                const bot = data.bot;
                
                // Set form values
                document.getElementById('bot-id').value = bot.id;
                document.getElementById('bot-name').value = bot.name;
                document.getElementById('bot-description').value = bot.description || '';
                document.getElementById('bot-greeting').value = bot.greeting || '';
                document.getElementById('bot-system-prompt').value = bot.system_prompt || '';
                
                // Set model selections with improved selection logic
                function selectModelSafely(selectElement, modelName) {
                    if (!selectElement || !modelName) return;
                    
                    // Try to find the exact match first
                    const exactMatch = Array.from(selectElement.options).find(option => 
                        option.value.toLowerCase() === modelName.toLowerCase()
                    );
                    
                    if (exactMatch) {
                        selectElement.value = exactMatch.value;
                        return;
                    }
                    
                    // If no exact match, try partial match
                    const partialMatch = Array.from(selectElement.options).find(option => 
                        option.value.toLowerCase().includes(modelName.toLowerCase())
                    );
                    
                    if (partialMatch) {
                        selectElement.value = partialMatch.value;
                        return;
                    }
                    
                    // If no match found, add a new option
                    const newOption = document.createElement('option');
                    newOption.value = modelName;
                    newOption.textContent = `${modelName} (Custom)`;
                    selectElement.appendChild(newOption);
                    selectElement.value = modelName;
                }
                
                // Ensure dropdowns are populated before setting values
                const baseModelSelect = document.getElementById('bot-base-model');
                const embeddingModelSelect = document.getElementById('bot-embedding-model');
                
                // Wait for dropdowns to be populated
                function checkAndSetModels() {
                    if (baseModelSelect.options.length > 1 && embeddingModelSelect.options.length > 1) {
                        if (bot.base_model) {
                            selectModelSafely(baseModelSelect, bot.base_model);
                        }
                        if (bot.embedding_model) {
                            selectModelSafely(embeddingModelSelect, bot.embedding_model);
                        }
                    } else {
                        // If dropdowns are not populated, try again after a short delay
                        setTimeout(checkAndSetModels, 100);
                    }
                }
                
                checkAndSetModels();
                
                // Set parameters
                if (bot.parameters) {
                    document.getElementById('bot-temperature').value = bot.parameters.temperature || 0.7;
                    document.getElementById('temperature-value').textContent = bot.parameters.temperature || 0.7;
                    
                    document.getElementById('bot-top-p').value = bot.parameters.top_p || 0.9;
                    document.getElementById('top-p-value').textContent = bot.parameters.top_p || 0.9;
                    
                    document.getElementById('bot-max-tokens').value = bot.parameters.max_tokens || 2048;
                }
                
                // Populate knowledge files
                if (bot.knowledge_files && bot.knowledge_files.length > 0) {
                    const filesListContainer = document.getElementById('knowledge-files-list');
                    filesListContainer.innerHTML = '';
                    
                    bot.knowledge_files.forEach(filename => {
                        const fileItem = createKnowledgeFileItem(filename, bot.id);
                        filesListContainer.appendChild(fileItem);
                    });
                }
            } else {
                showNotification(`Error loading bot: ${data.message}`, 'error');
            }
        } catch (error) {
            console.error('Error loading bot:', error);
            showNotification('Error loading bot details', 'error');
        }
    }
}

// Show bot list
function showBotList() {
    const botListSection = document.getElementById('bot-list-section');
    const botEditorSection = document.getElementById('bot-editor-section');
    const botModalTitle = document.getElementById('bot-modal-title');
    const createNewBotButton = document.getElementById('create-new-bot-button');
    const backToListButton = document.getElementById('back-to-bot-list-button');
    
    // Show list section, hide editor section
    botEditorSection.style.display = 'none';
    botListSection.style.display = 'block';
    
    // Update header
    botModalTitle.textContent = 'Bot Management';
    createNewBotButton.style.display = 'inline-block';
    backToListButton.style.display = 'none';
    
    // Reload bots
    loadBots();
}

// Save bot
async function saveBot() {
    try {
        // Get values from UI
        const botId = document.getElementById('bot-id').value;
        const botName = document.getElementById('bot-name').value;
        const botDescription = document.getElementById('bot-description').value;
        const botGreeting = document.getElementById('bot-greeting').value;
        const botBaseModel = document.getElementById('bot-base-model').value;
        const botEmbeddingModel = document.getElementById('bot-embedding-model').value;
        const botSystemPrompt = document.getElementById('bot-system-prompt').value;
        const botTemperature = parseFloat(document.getElementById('bot-temperature').value);
        const botTopP = parseFloat(document.getElementById('bot-top-p').value);
        const botMaxTokens = parseInt(document.getElementById('bot-max-tokens').value);
        
        // Validate required fields
        if (!botName || !botBaseModel || !botEmbeddingModel) {
            showNotification('Please fill in all required fields', 'error');
            return;
        }
        
        // Prepare bot data
        const botData = {
            name: botName,
            description: botDescription,
            greeting: botGreeting,
            base_model: botBaseModel,
            embedding_model: botEmbeddingModel,
            system_prompt: botSystemPrompt,
            parameters: {
                temperature: botTemperature,
                top_p: botTopP,
                max_tokens: botMaxTokens
            }
        };
        
        let response;
        let data;
        
        if (botId) {
            // Update existing bot
            response = await fetch(`/api/bots/${botId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(botData)
            });
            data = await response.json();
            
            if (data.status === 'success') {
                showNotification('Bot updated successfully');
                loadBots();
                showBotList();
            } else {
                showNotification(`Error updating bot: ${data.message}`, 'error');
            }
        } else {
            // Create new bot
            response = await fetch('/api/bots', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(botData)
            });
            
            data = await response.json();
            
            if (data.status === 'success') {
                showNotification('Bot created successfully');
                loadBots();
                showBotList();
            } else {
                showNotification(`Error creating bot: ${data.message}`, 'error');
            }
        }
    } catch (error) {
        console.error(`Error ${botId ? 'updating' : 'creating'} bot:`, error);
        showNotification(`Error ${botId ? 'updating' : 'creating'} bot`, 'error');
    }
}

// Edit bot
function editBot(botId) {
    showBotEditor(botId);
}

// Confirm delete bot
function confirmDeleteBot(botId) {
    if (confirm('Are you sure you want to delete this bot? This action cannot be undone.')) {
        deleteBot(botId)
            .then(response => {
                if (response.status === 'success') {
                    showNotification('Bot deleted successfully');
                    // Reload the bot list to update the UI immediately
                    loadBots();
                } else {
                    showNotification(`Error deleting bot: ${response.message}`, 'error');
                }
            })
            .catch(error => {
                console.error('Error deleting bot:', error);
                showNotification('Error deleting bot', 'error');
            });
    }
}

// Delete bot
async function deleteBot(botId) {
    try {
        const response = await fetch(`/api/bots/${botId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete bot');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Upload knowledge files
async function reindexKnowledgeFiles(botId) {
    if (!botId) {
        showNotification('Error', 'Bot ID is required', 'error');
        return;
    }

    // Show upload progress container
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const progressText = document.getElementById('upload-progress-text');
    const processingDetails = document.getElementById('processing-details');
    
    if (progressContainer) {
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = 'Starting re-indexing...';
        processingDetails.innerHTML = '<p>Re-indexing knowledge files. This may take some time for large files.</p>';
    }

    try {
        // Start the re-indexing process
        const response = await fetch(`/api/bots/${botId}/knowledge/reindex`, {
            method: 'POST'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || 'Failed to start re-indexing');
        }

        const responseData = await response.json();
        
        // After successful start, connect to SSE endpoint to get processing progress
        const eventSource = new EventSource(`/api/bots/${botId}/knowledge/progress`);
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.status === 'processing') {
                // Update progress bar
                const percent = data.progress * 100;
                progressBar.style.width = `${percent}%`;
                progressText.textContent = `Re-indexing: ${Math.round(percent)}%`;
                
                // Update processing details
                let detailsHtml = '';
                
                if (data.current_file) {
                    detailsHtml += `<p><strong>Current file:</strong> ${data.current_file}</p>`;
                }
                
                if (data.files_processed !== undefined) {
                    detailsHtml += `<p><strong>Files processed:</strong> ${data.files_processed}/${data.total_files || 0}</p>`;
                }
                
                if (data.chunks_processed !== undefined) {
                    detailsHtml += `<p><strong>Chunks processed:</strong> ${data.chunks_processed}/${data.total_chunks || 0}</p>`;
                }
                
                processingDetails.innerHTML = detailsHtml || '<p>Processing...</p>';
            } else if (data.status === 'complete') {
                // Processing complete
                progressBar.style.width = '100%';
                progressText.textContent = 'Re-indexing complete!';
                
                let summaryHtml = '';
                if (data.files_processed !== undefined) {
                    summaryHtml += `<p><strong>Files processed:</strong> ${data.files_processed}</p>`;
                }
                
                if (data.total_chunks !== undefined) {
                    summaryHtml += `<p><strong>Total chunks:</strong> ${data.total_chunks}</p>`;
                }
                
                if (data.processing_time !== undefined) {
                    summaryHtml += `<p><strong>Processing time:</strong> ${data.processing_time.toFixed(1)}s</p>`;
                }
                
                processingDetails.innerHTML = summaryHtml || '<p>Processing complete!</p>';
                
                // Close the event source
                eventSource.close();
                
                // Show success notification
                showNotification('Success', 'Knowledge base re-indexed successfully', 'success');
                
                // Hide progress after a delay
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 5000);
            } else if (data.status === 'error') {
                // Error during processing
                progressText.textContent = 'Error during re-indexing';
                processingDetails.innerHTML = `<p class="error">${data.error || data.message || 'Unknown error'}</p>`;
                
                // Close the event source
                eventSource.close();
                
                // Show error notification
                showNotification('Error', data.error || data.message || 'Unknown error', 'error');
            }
        };
        
        eventSource.onerror = function() {
            eventSource.close();
            progressText.textContent = 'Lost connection to server';
            showNotification('Warning', 'Lost connection to progress updates', 'warning');
        };
        
        // Show initial success notification
        showNotification('Info', responseData.message, 'info');
        
    } catch (error) {
        console.error('Error starting re-indexing:', error);
        showNotification('Error', error.message, 'error');
        
        if (progressContainer) {
            progressText.textContent = 'Re-indexing failed';
            processingDetails.innerHTML = `<p class="error">${error.message}</p>`;
        }
    }
}

// Upload knowledge files
async function uploadKnowledgeFiles(botId, files) {
    if (!botId || !files || files.length === 0) {
        showNotification('No files selected', 'error');
        return;
    }

    // Validate files
    const validFiles = Array.from(files).filter(file => 
        file.size > 0 && 
        file.size <= 100 * 1024 * 1024 && // 100MB max
        ['.txt', '.pdf', '.md', '.xml', '.json', '.csv', '.xls', '.xlsx']
            .some(ext => file.name.toLowerCase().endsWith(ext))
    );

    if (validFiles.length === 0) {
        showNotification('No valid files to upload. Supported formats: .txt, .pdf, .md, .xml, .json, .csv, .xls, .xlsx (max 100MB)', 'error');
        return;
    }

    const formData = new FormData();
    validFiles.forEach(file => {
        formData.append('files', file);
    });

    // Initialize progress UI
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = progressContainer?.querySelector('.progress-bar');
    const progressText = progressContainer?.querySelector('.progress-status');
    const processingDetails = progressContainer?.querySelector('.progress-details');
    
    if (progressContainer) {
        progressContainer.style.display = 'block';
        if (progressBar) progressBar.style.width = '0%';
        if (progressText) progressText.textContent = 'Starting upload...';
        if (processingDetails) processingDetails.innerHTML = '';
    }

    try {
        // Show processing UI
        if (progressBar) progressBar.style.width = '10%';
        if (progressText) progressText.textContent = 'Preparing upload...';

        // First, upload the files
        const response = await fetch(`/api/bots/${botId}/knowledge`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let errorMsg = 'Failed to upload files';
            try {
                const errorData = await response.json();
                errorMsg = errorData.message || errorMsg;
            } catch (e) {
                console.error('Error parsing error response:', e);
            }
            throw new Error(errorMsg);
        }
        
        // Update progress bar to 50% for upload complete
        if (progressBar) progressBar.style.width = '50%';
        if (progressText) progressText.textContent = 'Processing documents...';
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update progress with processing stats
            if (data.processing_stats) {
                const stats = data.processing_stats;
                
                // Update progress bar to 100%
                if (progressBar) progressBar.style.width = '100%';
                if (progressText) progressText.textContent = 'Processing complete!';
                
                // Update stats if elements exist
                if (progressContainer) {
                    const chunksProcessed = progressContainer.querySelector('.chunks-processed');
                    const processingTime = progressContainer.querySelector('.processing-time');
                    if (chunksProcessed) chunksProcessed.textContent = stats.total_chunks || 0;
                    if (processingTime) processingTime.textContent = (stats.processing_time || 0).toFixed(1);
                }
                
                // Update file progress items if file stats exist
                if (stats.file_stats?.length > 0) {
                    const fileList = progressContainer?.querySelector('.file-progress-list');
                    if (fileList) {
                        fileList.innerHTML = '';
                        
                        stats.file_stats.forEach(fileStat => {
                            const fileItem = document.createElement('div');
                            fileItem.className = 'file-progress-item';
                            fileItem.innerHTML = `
                                <div class="file-progress-name">${fileStat.filename || 'Unknown file'}</div>
                                <div class="file-progress-details">
                                    <span>Size: ${fileStat.size_kb ? fileStat.size_kb.toFixed(1) + ' KB' : 'N/A'}</span>
                                    <span>Chunks: ${fileStat.chunks || 0}</span>
                                    <span>Time: ${fileStat.processing_time ? fileStat.processing_time.toFixed(1) + 's' : 'N/A'}</span>
                                </div>
                            `;
                            fileList.appendChild(fileItem);
                        });
                    }
                }
                
                // Hide progress after 5 seconds
                if (progressContainer) {
                    progressContainer.classList.add('active');
                    setTimeout(() => {
                        progressContainer.style.display = 'none';
                        progressContainer.classList.remove('active');
                    }, 5000);
                }
            }
            
            showNotification('Knowledge files uploaded and processed successfully');
            
            // Update knowledge files list if container exists
            const filesListContainer = document.getElementById('knowledge-files-list');
            if (filesListContainer && data.bot?.knowledge_files) {
                filesListContainer.innerHTML = '';
                
                data.bot.knowledge_files.forEach(filename => {
                    const fileItem = createKnowledgeFileItem(filename, botId);
                    if (fileItem) {
                        filesListContainer.appendChild(fileItem);
                    }
                });
            }
            
            // Clear file input
            const fileInput = document.getElementById('knowledge-file-input');
            if (fileInput) fileInput.value = '';
        } else {
            throw new Error(data.message || 'Unknown error processing files');
        }
    } catch (error) {
        let errorMsg = 'Error uploading files';
        
        // Handle 413 payload too large
        if (error.message && error.message.includes('413')) {
            errorMsg = 'File too large (max 100MB)';
        } 
        // Handle HTML error responses (when server returns HTML error page)
        else if (error instanceof SyntaxError && error.message.includes('Unexpected token')) {
            errorMsg = 'Server rejected upload - file may be too large';
        }
        
        console.error('Upload error:', error);
        showNotification(errorMsg, 'error');
    }
}

// Create knowledge file item
function createKnowledgeFileItem(filename, botId) {
    const fileItem = document.createElement('div');
    fileItem.className = 'knowledge-file-item';
    
    const fileName = document.createElement('div');
    fileName.className = 'knowledge-file-name';
    fileName.textContent = filename;
    
    const removeButton = document.createElement('button');
    removeButton.className = 'knowledge-file-remove';
    removeButton.innerHTML = '<i class="fas fa-times"></i>';
    removeButton.title = 'Remove file';
    removeButton.addEventListener('click', () => removeKnowledgeFile(botId, filename));
    
    fileItem.appendChild(fileName);
    fileItem.appendChild(removeButton);
    
    return fileItem;
}

// Remove knowledge file
async function removeKnowledgeFile(botId, filename) {
    try {
        const response = await fetch(`/api/bots/${botId}/knowledge/${filename}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification('Knowledge file removed successfully');
            
            // Update knowledge files list
            const filesListContainer = document.getElementById('knowledge-files-list');
            filesListContainer.innerHTML = '';
            
            data.bot.knowledge_files.forEach(filename => {
                const fileItem = createKnowledgeFileItem(filename, botId);
                filesListContainer.appendChild(fileItem);
            });
        } else {
            showNotification(`Error removing knowledge file: ${data.message}`, 'error');
        }
    } catch (error) {
        console.error('Error removing knowledge file:', error);
        showNotification('Error removing knowledge file', 'error');
    }
}

// Start chat with bot
function startChatWithBot(botId) {
    // Close the bot management modal
    document.getElementById('bot-management-modal').style.display = 'none';
    
    // TODO: Implement starting a chat with the selected bot
    showNotification('Bot chat functionality coming soon!');
}

// Create a new conversation
async function createNewConversation() {
    try {
        const model = modelSelect.value || currentModel;
        const now = new Date();
        const dateStr = now.toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        const title = `New Chat - ${dateStr}`;
        const response = await fetch('/api/conversations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, model, secret: secretChatMode })
        });
        const data = await response.json();
        if (data.status === 'success') {
            currentConversationId = data.conversation_id;
            currentConversationSecret = !!data.secret;
            loadConversation(currentConversationId);
            loadConversations(true);
            isNewConversation = true;
            // Show system message if secret mode is on
            if (secretChatMode) addSystemSecretChatMessage();
            else removeSystemSecretChatMessage();
            // Hide landing page when creating a new conversation
            const landingPage = document.getElementById('landing-page');
            if (landingPage) {
                landingPage.style.display = 'none';
            }
            return currentConversationId;
        } else {
            addMessage(`Error: ${data.message}`, false, 'system');
            return null;
        }
    } catch (error) {
        addMessage('Failed to create conversation', false, 'system');
        console.error('Error creating conversation:', error);
        return null;
    }
}

// --- Secret Chat Mode State ---
let secretChatMode = false;
let currentConversationSecret = false;

function toggleSecretChatMode() {
    secretChatMode = !secretChatMode;
    const secretButton = document.getElementById('toggle-secret-chat');
    const chatTitle = document.getElementById('chat-title');
    const chatContainer = document.getElementById('chat-container');
    if (secretChatMode) {
        // Always clear chat content and conversation state when entering secret chat
        currentConversationId = null;
        if (chatContainer) chatContainer.innerHTML = '';
        if (chatTitle) chatTitle.textContent = 'Secret Chat';
        secretButton.classList.add('secret-active');
        addSystemSecretChatMessage();
    } else {
        secretButton.classList.remove('secret-active');
        // Fade out all messages from the chat container when secret mode is toggled off
        if (chatContainer) {
            const messages = Array.from(chatContainer.children);
            if (messages.length > 0) {
                messages.forEach((msg, idx) => {
                    msg.classList.add('fade-to-ashes');
                    setTimeout(() => {
                        if (msg.parentNode) msg.parentNode.removeChild(msg);
                        // After last message, show system message
                        if (idx === messages.length - 1) {
                            addMessage('Secret chat messages have faded into ashes and are gone forever.', false, 'system');
                        }
                    }, 1100);
                });
            } else {
                // If no messages, still show system message
                addMessage('Secret chat messages have faded into ashes and are gone forever.', false, 'system');
            }
        }
        // Restore chat title to default
        if (chatTitle) chatTitle.textContent = 'New Chat';
    }
    // Existing logic (system message etc.) remains unchanged
    updateSecretChatUI();
}

// Ensure secret indicator is hidden on load
window.addEventListener('DOMContentLoaded', function() {
    const indicator = document.getElementById('secret-chat-indicator');
    if (indicator) indicator.style.display = 'none';
    const btn = document.getElementById('toggle-secret-chat');
    if (btn) btn.classList.remove('secret-active');
});

// --- Animated Loading Dots for "MIDAS is doing AI magic..." ---
let aiMagicDotsInterval = null;
function startAiMagicDotsAnimation() {
    // Find the spinner message element
    const spinner = document.querySelector('.image-loading-spinner');
    if (!spinner) return;
    let baseText = 'MIDAS is doing AI magic';
    let dotCount = 0;
    // Find the text node inside spinner
    let textNode = Array.from(spinner.childNodes).find(n => n.nodeType === Node.TEXT_NODE || (n.nodeType === Node.ELEMENT_NODE && n.nodeName === 'SPAN'));
    if (!textNode) {
        // fallback: create a span
        textNode = document.createElement('span');
        spinner.appendChild(textNode);
    }
    function updateDots() {
        dotCount = (dotCount + 1) % 4;
        let dots = '.'.repeat(dotCount);
        textNode.textContent = `${baseText}${dots}`;
    }
    // Set initial
    textNode.textContent = `${baseText}...`;
    aiMagicDotsInterval = setInterval(updateDots, 500);
}
function stopAiMagicDotsAnimation() {
    if (aiMagicDotsInterval) {
        clearInterval(aiMagicDotsInterval);
        aiMagicDotsInterval = null;
    }
}

// Show full image overlay
function showImageOverlay(imgSrc, altText = '') {
    // Remove existing overlay if present
    const existingOverlay = document.getElementById('image-overlay-modal');
    if (existingOverlay) existingOverlay.remove();

    // Create overlay elements
    const overlay = document.createElement('div');
    overlay.id = 'image-overlay-modal';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.right = '0';
    overlay.style.bottom = '0';
    overlay.style.background = 'rgba(0,0,0,0.8)';
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.zIndex = '9999';

    // Create image
    const img = document.createElement('img');
    img.src = imgSrc;
    img.alt = altText;
    img.style.maxWidth = '90vw';
    img.style.maxHeight = '90vh';
    img.style.borderRadius = '10px';
    img.style.boxShadow = '0 4px 32px rgba(0,0,0,0.5)';
    img.style.background = '#222';

    // Close on click outside image or on ESC
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.remove();
    });
    window.addEventListener('keydown', function escListener(e) {
        if (e.key === 'Escape') {
            overlay.remove();
            window.removeEventListener('keydown', escListener);
        }
    });
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    closeBtn.style.position = 'absolute';
    closeBtn.style.top = '32px';
    closeBtn.style.right = '48px';
    closeBtn.style.fontSize = '2.2em';
    closeBtn.style.background = 'transparent';
    closeBtn.style.color = '#fff';
    closeBtn.style.border = 'none';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.zIndex = '10000';
    closeBtn.addEventListener('click', () => overlay.remove());

    overlay.appendChild(img);
    overlay.appendChild(closeBtn);
    document.body.appendChild(overlay);
}

// Attach overlay to generated images and full screen button
function enableImageOverlay(imgElement) {
    if (!imgElement) return;
    imgElement.style.cursor = 'zoom-in';
    imgElement.addEventListener('click', function() {
        showImageOverlay(imgElement.src, imgElement.alt);
    });
}

// Patch 1: When displaying generated image
// Find code that creates img element with class 'generated-image'
// After: imageContainer.appendChild(img);
// Add: enableImageOverlay(img);

// Patch 2: For full screen button, call showImageOverlay with the image src
// Find code that creates the full screen button and add:
// fullscreenBtn.addEventListener('click', () => showImageOverlay(img.src, img.alt));

// CSS for image spinner and download button (inject if not present)
(function injectImageGenCSS() {
    if (!document.getElementById('image-gen-css')) {
        const style = document.createElement('style');
        style.id = 'image-gen-css';
        style.innerHTML = `
        .image-loading-spinner { display: flex; align-items: center; gap: 12px; margin: 16px 0; }
        .image-loading-spinner .xeno-loader {
            width: 48px;
            height: 48px;
            animation: float 2s ease-in-out infinite;
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        .generated-image { 
            border-radius: 8px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
            margin: 12px 0;
            max-width: 100%;
            cursor: pointer;
        }
        .generated-image-container {
            position: relative;
            display: inline-block;
        }
        .image-actions {
            display: flex;
            gap: 8px;
            margin-top: 4px;
        }
        .image-actions button {
            background: transparent;
            border: none;
            color: var(--text-color-muted);
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            transition: background-color 0.2s, color 0.2s;
        }
        .image-actions button:hover {
            background-color: var(--hover-color);
            color: var(--text-color);
        }
        .error-message { color: #c00; font-weight: bold; }
        `;
        document.head.appendChild(style);
    }
})();

function addSystemSecretChatMessage() {
    // Only add if not already present
    if (!document.getElementById('system-secret-chat-msg')) {
        const chatContainer = document.getElementById('chat-container');
        const div = document.createElement('div');
        div.className = 'system-message';
        div.id = 'system-secret-chat-msg';
        div.innerHTML = '<i class="fas fa-user-secret"></i> Secret Chat is enabled. Messages will not be saved.';
        chatContainer.insertBefore(div, chatContainer.firstChild);
    }
}

function removeSystemSecretChatMessage() {
    const msg = document.getElementById('system-secret-chat-msg');
    if (msg) msg.remove();
}

function toggleSecretChatMode() {
    secretChatMode = !secretChatMode;
    const secretButton = document.getElementById('toggle-secret-chat');
    const chatTitle = document.getElementById('chat-title');
    const chatContainer = document.getElementById('chat-container');
    if (secretChatMode) {
        // Always clear chat content and conversation state when entering secret chat
        currentConversationId = null;
        if (chatContainer) chatContainer.innerHTML = '';
        if (chatTitle) chatTitle.textContent = 'Secret Chat';
        secretButton.classList.add('secret-active');
        addSystemSecretChatMessage();
    } else {
        secretButton.classList.remove('secret-active');
        // Fade out all messages from the chat container when secret mode is toggled off
        if (chatContainer) {
            const messages = Array.from(chatContainer.children);
            if (messages.length > 0) {
                messages.forEach((msg, idx) => {
                    msg.classList.add('fade-to-ashes');
                    setTimeout(() => {
                        if (msg.parentNode) msg.parentNode.removeChild(msg);
                        // After last message, show system message
                        if (idx === messages.length - 1) {
                            addMessage('Secret chat messages have faded into ashes and are gone forever.', false, 'system');
                        }
                    }, 1100);
                });
            } else {
                // If no messages, still show system message
                addMessage('Secret chat messages have faded into ashes and are gone forever.', false, 'system');
            }
        }
        // Restore chat title to default
        if (chatTitle) chatTitle.textContent = 'New Chat';
    }
    // Existing logic (system message etc.) remains unchanged
    updateSecretChatUI();
}

// Utility function to hide the landing page
function hideLandingPage() {
    const landingPage = document.getElementById('landing-page');
    if (landingPage) {
        landingPage.style.display = 'none';
    }
}

// Mobile responsiveness enhancements
function setupMobileResponsiveness() {
    const sidebar = document.querySelector('.sidebar');
    const chatPanel = document.querySelector('.chat-panel');
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    
    // Function to check if we're on mobile
    function isMobile() {
        return window.innerWidth <= 768;
    }
    
    // Handle window resize
    window.addEventListener('resize', function() {
        if (isMobile()) {
            // On mobile, always collapse sidebar initially
            if (sidebar && !sidebar.classList.contains('sidebar-collapsed')) {
                sidebar.classList.add('sidebar-collapsed');
            }
            if (chatPanel && !chatPanel.classList.contains('sidebar-collapsed')) {
                chatPanel.classList.add('sidebar-collapsed');
            }
        }
    });
    
    // Initial check on page load
    if (isMobile()) {
        if (sidebar) sidebar.classList.add('sidebar-collapsed');
        if (chatPanel) chatPanel.classList.add('sidebar-collapsed');
    }
    
    // Enhance sidebar toggle for mobile
    if (sidebarToggle) {
        const originalClickHandler = sidebarToggle.onclick;
        sidebarToggle.onclick = function(e) {
            if (originalClickHandler) {
                originalClickHandler.call(this, e);
            }
            
            // On mobile, when sidebar is opened, add an overlay to allow closing by tapping outside
            if (isMobile() && sidebar && !sidebar.classList.contains('sidebar-collapsed')) {
                const overlay = document.createElement('div');
                overlay.className = 'mobile-sidebar-overlay';
                document.body.appendChild(overlay);
                
                overlay.addEventListener('click', function() {
                    sidebar.classList.add('sidebar-collapsed');
                    chatPanel.classList.add('sidebar-collapsed');
                    document.body.removeChild(overlay);
                });
            } else {
                // Remove overlay when sidebar is closed
                const overlay = document.querySelector('.mobile-sidebar-overlay');
                if (overlay) {
                    document.body.removeChild(overlay);
                }
            }
        };
    }
    
    // Enhance auth modal for mobile
    const authModal = document.getElementById('auth-modal');
    if (authModal) {
        // Make sure auth modal is properly centered on mobile
        function adjustAuthModal() {
            if (isMobile()) {
                const modalContent = authModal.querySelector('.modal-content');
                if (modalContent) {
                    modalContent.style.maxHeight = (window.innerHeight * 0.9) + 'px';
                    modalContent.style.overflow = 'auto';
                }
            }
        }
        
        window.addEventListener('resize', adjustAuthModal);
        // Also adjust when modal is shown
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.attributeName === 'style' && 
                    authModal.style.display !== 'none') {
                    adjustAuthModal();
                }
            });
        });
        
        observer.observe(authModal, { attributes: true });
    }
}

// Mobile header functionality
function setupMobileHeader() {
    const mobileMessagesButton = document.getElementById('mobile-messages-button');
    const messagesModal = document.getElementById('messages-modal');
    const closeMessagesModal = document.getElementById('close-messages-modal');
    const sidebar = document.querySelector('.sidebar');
    const chatPanel = document.querySelector('.chat-panel');
    
    // Function to check if we're on mobile
    function isMobile() {
        return window.innerWidth <= 768;
    }
    
    // Handle window resize
    window.addEventListener('resize', function() {
        if (isMobile()) {
            // On mobile, always collapse sidebar initially
            if (sidebar && !sidebar.classList.contains('sidebar-collapsed')) {
                sidebar.classList.add('sidebar-collapsed');
            }
            if (chatPanel && !chatPanel.classList.contains('sidebar-collapsed')) {
                chatPanel.classList.add('sidebar-collapsed');
            }
        }
    });
    
    // Initial check on page load
    if (isMobile()) {
        if (sidebar) sidebar.classList.add('sidebar-collapsed');
        if (chatPanel) chatPanel.classList.add('sidebar-collapsed');
    }
    
    // Handle mobile messages button click
    if (mobileMessagesButton) {
        mobileMessagesButton.addEventListener('click', function() {
            // Show messages modal
            showMessagesModal();
        });
    }
    
    // Function to show messages modal
    function showMessagesModal() {
        if (!messagesModal) return;
        
        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'mobile-modal-overlay';
        document.body.appendChild(overlay);
        
        // Show modal
        messagesModal.style.display = 'block';
        
        // Populate messages list from conversation list
        populateMessagesModal();
        
        // Close when clicking overlay
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                hideMessagesModal();
            }
        });
    }
    
    // Function to hide messages modal
    function hideMessagesModal() {
        if (!messagesModal) return;
        
        // Hide modal
        messagesModal.style.display = 'none';
        
        // Remove overlay
        const overlay = document.querySelector('.mobile-modal-overlay');
        if (overlay) {
            document.body.removeChild(overlay);
        }
    }
    
    // Close button for messages modal
    if (closeMessagesModal) {
        closeMessagesModal.addEventListener('click', hideMessagesModal);
    }
    
    // Function to populate messages modal with conversations
    function populateMessagesModal() {
        const messagesList = document.querySelector('.messages-list');
        const conversationList = document.getElementById('conversation-list');
        
        if (!messagesList || !conversationList) return;
        
        // Clear existing items
        messagesList.innerHTML = '';
        
        // Clone conversation items to messages list
        const conversationItems = conversationList.querySelectorAll('.conversation-item');
        conversationItems.forEach(item => {
            const clone = item.cloneNode(true);
            
            // Add click handler to select conversation
            clone.addEventListener('click', function() {
                // Get conversation ID
                const conversationId = clone.getAttribute('data-id');
                if (conversationId) {
                    // Load the conversation
                    loadConversation(conversationId);
                    
                    // Hide the modal
                    hideMessagesModal();
                }
            });
            
            messagesList.appendChild(clone);
        });
        
        // If no conversations, show a message
        if (conversationItems.length === 0) {
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-state';
            emptyState.style.padding = '2rem';
            emptyState.style.textAlign = 'center';
            emptyState.style.color = 'var(--text-secondary)';
            emptyState.innerHTML = '<i class="fas fa-comments" style="font-size: 2rem; margin-bottom: 1rem;"></i><p>No conversations yet</p>';
            messagesList.appendChild(emptyState);
        }
    }
    
    // Search functionality for messages modal
    const messagesSearchInput = document.getElementById('messages-search-input');
    if (messagesSearchInput) {
        messagesSearchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const conversationItems = document.querySelectorAll('.messages-list .conversation-item');
            
            conversationItems.forEach(item => {
                const title = item.querySelector('.conversation-title');
                const titleText = title ? title.textContent.toLowerCase() : '';
                
                if (titleText.includes(searchTerm)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
    
    // Adjust header based on screen size
    function adjustHeaderForMobile() {
        const chatInfo = document.querySelector('.chat-info');
        const modelSelector = document.querySelector('.model-selector');
        const mobileLogo = document.querySelector('.mobile-logo');
        
        if (isMobile()) {
            // On mobile, always collapse sidebar initially
            if (chatInfo) chatInfo.style.display = 'none';
            if (modelSelector) modelSelector.style.display = 'none';
            if (mobileLogo) mobileLogo.style.display = 'flex';
            if (mobileMessagesButton) mobileMessagesButton.style.display = 'flex';
        } else {
            // Desktop view
            if (chatInfo) chatInfo.style.display = 'flex';
            if (modelSelector) modelSelector.style.display = 'flex';
            if (mobileLogo) mobileLogo.style.display = 'none';
            if (mobileMessagesButton) mobileMessagesButton.style.display = 'none';
        }
    }
    
    // Call on page load and window resize
    adjustHeaderForMobile();
    window.addEventListener('resize', adjustHeaderForMobile);
}

// Utility function to adjust the message input textarea height
function adjustTextareaHeight() {
    const textarea = document.getElementById('message-input');
    if (textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = (textarea.scrollHeight) + 'px';
    }
}

// Show the landing page
function showLandingPage() {
    const landingPage = document.getElementById('landing-page');
    const chatContainer = document.getElementById('chat-container');
    
    if (!landingPage) return;
    
    // Clear any existing messages except the landing page
    Array.from(chatContainer.children).forEach(child => {
        if (child.id !== 'landing-page' && !child.classList.contains('system-message')) {
            child.remove();
        }
    });
    
    // Initially set opacity to 0 for fade-in effect
    landingPage.style.opacity = '0';
    landingPage.style.transition = 'opacity 0.5s ease-in-out';
    
    // Show the landing page
    landingPage.style.display = 'flex';
    
    // Trigger fade-in effect after a small delay to ensure display is applied first
    setTimeout(() => {
        landingPage.style.opacity = '1';
    }, 50);
    
    // Populate prompt suggestions
    const suggestionsContainer = landingPage.querySelector('.prompt-suggestions');
    if (suggestionsContainer) {
        suggestionsContainer.innerHTML = '';
        
        // Get available models
        const availableModelIds = Array.from(modelSelect.options).map(option => option.value);
        console.log('Available models for prompt suggestions:', availableModelIds);
        
        // Filter suggestions
        let filteredSuggestions = promptSuggestions.filter(suggestion => {
            // If no specific model is required, include it
            if (!suggestion.model) return true;
            
            // For workflow models, check if any workflow is available
            if (suggestion.model.startsWith('workflow:')) {
                return workflowModels.length > 0;
            }
            
            // For regular models, check if it's in available models
            return availableModelIds.includes(suggestion.model);
        });
        
        // Ensure we have at least 8 suggestions if possible
        if (filteredSuggestions.length < 8) {
            // Add generic suggestions that don't require specific models
            const genericSuggestions = [
                {
                    title: "Creative Writing",
                    prompt: "Write a short story about a detective solving a mystery in a small town.",
                    model: "deepseek-r1:7b",
                    category: "writing"
                },
                {
                    title: "Coding Help",
                    prompt: "How do I implement a binary search algorithm?",
                    model: "codellama:latest",
                    category: "coding"
                },
                {
                    title: "Travel Planning",
                    prompt: "Create a 3-day itinerary for visiting Tokyo, Japan.",
                    model: "mistral:latest",
                    category: "travel"
                },
                {
                    title: "Philosophy Discussion",
                    prompt: "Discuss the trolley problem and its ethical implications.",
                    model: "llama3.1:8b",
                    category: "philosophy"
                },
                {
                    title: "Recipe Creation",
                    prompt: "Create a recipe for a vegetarian dinner that's quick to prepare.",
                    model: "phi3.5:latest",
                    category: "cooking"
                },
                {
                    title: "Language Learning",
                    prompt: "Teach me 10 common phrases in Spanish with their pronunciations.",
                    model: "deepseek-r1:7b",
                    category: "language"
                }
            ];
            
            // Add generic suggestions until we have 8
            for (let i = 0; i < genericSuggestions.length && filteredSuggestions.length < 8; i++) {
                if (!filteredSuggestions.some(s => s.title === genericSuggestions[i].title)) {
                    filteredSuggestions.push(genericSuggestions[i]);
                }
            }
        }
        
        // Always show exactly 8 suggestions if possible
        filteredSuggestions = filteredSuggestions.slice(0, 8);
        
        console.log(`Displaying ${filteredSuggestions.length} prompt suggestions`);
        
        // Create cards for each suggestion
        filteredSuggestions.forEach(suggestion => {
            const card = document.createElement('div');
            card.className = 'prompt-card';
            card.dataset.prompt = suggestion.prompt;
            card.dataset.model = suggestion.model;
            
            card.innerHTML = `
                <h3>${suggestion.title}</h3>
                <p>${suggestion.prompt}</p>
                <div class="prompt-model">model: ${suggestion.model || 'Any'}</div>
            `;
            
            card.addEventListener('click', () => {
                usePromptSuggestion(suggestion.prompt, suggestion.model);
            });
            
            suggestionsContainer.appendChild(card);
        });
        
        // Initialize carousel navigation
        initCarouselNavigation();
        
        // Trigger fade-in animation after a short delay
        setTimeout(() => {
            landingPage.classList.add('visible');
        }, 100);
    }
}

// Function to initialize carousel navigation
function initCarouselNavigation() {
    const landingPage = document.getElementById('landing-page');
    const container = landingPage.querySelector('.prompt-suggestions');
    const prevBtn = document.querySelector('.carousel-button.prev');
    const nextBtn = document.querySelector('.carousel-button.next');
    
    if (!container || !prevBtn || !nextBtn) return;
    
    // Calculate the scroll amount (width of one card + gap)
    const scrollAmount = 250 + 16; // card width + gap
    
    // Add click event to previous button
    prevBtn.addEventListener('click', () => {
        // Check if at the beginning
        if (container.scrollLeft <= 0) {
            // Jump to the end (loop around)
            container.scrollLeft = container.scrollWidth;
            // Then scroll back one card to create smooth transition
            setTimeout(() => {
                container.scrollBy({
                    left: -scrollAmount,
                    behavior: 'smooth'
                });
            }, 10);
        } else {
            // Normal scroll
            container.scrollBy({
                left: -scrollAmount,
                behavior: 'smooth'
            });
        }
    });
    
    // Add click event to next button
    nextBtn.addEventListener('click', () => {
        const maxScrollLeft = container.scrollWidth - container.clientWidth;
        
        // Check if at the end
        if (container.scrollLeft >= maxScrollLeft - 10) {
            // Jump to the beginning (loop around)
            container.scrollLeft = 0;
        } else {
            // Normal scroll
            container.scrollBy({
                left: scrollAmount,
                behavior: 'smooth'
            });
        }
    });
    
    // Always show navigation buttons since we're looping
    prevBtn.style.display = 'flex';
    nextBtn.style.display = 'flex';
    
    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
            prevBtn.click();
        } else if (e.key === 'ArrowRight') {
            nextBtn.click();
        }
    });
}

// Sample prompt suggestions for different models
const promptSuggestions = [
    {
        title: "Creative Writing",
        prompt: "Write a short story about a time traveler who accidentally changes history.",
        model: "llama3:8b",
        category: "writing"
    },
    {
        title: "Code Explanation",
        prompt: "Explain how async/await works in JavaScript with examples.",
        model: "mistral:7b",
        category: "coding"
    },
    {
        title: "Image Generation - Flux Enhanced",
        prompt: "A futuristic cityscape with flying cars and neon lights, digital art style.",
        model: "workflow:Flux Enhanced",
        category: "image"
    },
    {
        title: "Research Summary",
        prompt: "Summarize the latest developments in quantum computing.",
        model: "llama3:8b",
        category: "research"
    },
    {
        title: "Business Idea",
        prompt: "Generate a business plan for a sustainable food delivery service.",
        model: "gemma:7b",
        category: "business"
    },
    {
        title: "Data Analysis",
        prompt: "How would you analyze customer churn data to improve retention?",
        model: "mistral:7b",
        category: "data"
    },
    {
        title: "Coding Assistant",
        prompt: "Write a Python function that calculates the Fibonacci sequence recursively with memoization.",
        model: "llama2",
        category: "coding"
    },
    {
        title: "Learning Assistance",
        prompt: "Explain the concept of neural networks in simple terms.",
        model: "mistral:latest",
        category: "education"
    }
];

// Function to use a prompt suggestion
function usePromptSuggestion(prompt, modelId) {
    // Hide the landing page with animation
    hideLandingPage();
    
    // Set the prompt in the message input
    messageInput.value = prompt;
    
    // If a specific model is required, select it
    if (modelId) {
        // Find the option with this model ID
        const option = Array.from(modelSelect.options).find(opt => opt.value === modelId);
        if (option) {
            modelSelect.value = modelId;
            // Trigger change event to update any dependent UI
            modelSelect.dispatchEvent(new Event('change'));
        } else {
            console.warn(`Model ${modelId} not found in available models`);
        }
    }
    
    // Focus on the message input
    messageInput.focus();
    
    // Adjust the textarea height
    adjustTextareaHeight();
}

// Function to initialize carousel navigation
function initCarouselNavigation() {
    const container = document.querySelector('.prompt-suggestions');
    const prevBtn = document.querySelector('.carousel-button.prev');
    const nextBtn = document.querySelector('.carousel-button.next');
    
    if (!container || !prevBtn || !nextBtn) return;
    
    // Calculate the scroll amount (width of one card + gap)
    const scrollAmount = 250 + 16; // card width + gap
    
    // Add click event to previous button
    prevBtn.addEventListener('click', () => {
        // Check if at the beginning
        if (container.scrollLeft <= 0) {
            // Jump to the end (loop around)
            container.scrollLeft = container.scrollWidth;
            // Then scroll back one card to create smooth transition
            setTimeout(() => {
                container.scrollBy({
                    left: -scrollAmount,
                    behavior: 'smooth'
                });
            }, 10);
        } else {
            // Normal scroll
            container.scrollBy({
                left: -scrollAmount,
                behavior: 'smooth'
            });
        }
    });
    
    // Add click event to next button
    nextBtn.addEventListener('click', () => {
        const maxScrollLeft = container.scrollWidth - container.clientWidth;
        
        // Check if at the end
        if (container.scrollLeft >= maxScrollLeft - 10) {
            // Jump to the beginning (loop around)
            container.scrollLeft = 0;
        } else {
            // Normal scroll
            container.scrollBy({
                left: scrollAmount,
                behavior: 'smooth'
            });
        }
    });
    
    // Always show navigation buttons since we're looping
    prevBtn.style.display = 'flex';
    nextBtn.style.display = 'flex';
    
    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
            prevBtn.click();
        } else if (e.key === 'ArrowRight') {
            nextBtn.click();
        }
    });
}

// Synchronize the sidebar model selector with the header model selector
function initializeSidebarModelSelector() {
    const headerModelSelect = document.getElementById('model-select');
    const sidebarModelSelect = document.getElementById('sidebar-model-select');
    const headerRefreshButton = document.getElementById('refresh-models');
    const sidebarRefreshButton = document.getElementById('sidebar-refresh-models');
    
    if (!headerModelSelect || !sidebarModelSelect) return;
    
    // Initialize sidebar model select with the same options as header model select
    function syncModelOptions() {
        // Clear existing options in sidebar selector
        while (sidebarModelSelect.firstChild) {
            sidebarModelSelect.removeChild(sidebarModelSelect.firstChild);
        }
        
        // Copy options from header selector to sidebar selector
        Array.from(headerModelSelect.children).forEach(child => {
            const clone = child.cloneNode(true);
            sidebarModelSelect.appendChild(clone);
        });
    }
    
    // Sync the selected value between selectors
    function syncSelectedModel(sourceSelect, targetSelect) {
        if (sourceSelect && targetSelect) {
            targetSelect.value = sourceSelect.value;
        }
    }
    
    // Initial sync
    syncModelOptions();
    
    // When header model select changes, update sidebar model select
    headerModelSelect.addEventListener('change', () => {
        syncSelectedModel(headerModelSelect, sidebarModelSelect);
    });
    
    // When sidebar model select changes, update header model select and trigger change event
    sidebarModelSelect.addEventListener('change', () => {
        syncSelectedModel(sidebarModelSelect, headerModelSelect);
        // Trigger change event on header model select to ensure any listeners are notified
        const event = new Event('change');
        headerModelSelect.dispatchEvent(event);
    });
    
    // When models are refreshed from header button
    if (headerRefreshButton) {
        headerRefreshButton.addEventListener('click', () => {
            // After a short delay to allow models to load
            setTimeout(syncModelOptions, 500);
        });
    }
    
    // When models are refreshed from sidebar button
    if (sidebarRefreshButton) {
        sidebarRefreshButton.addEventListener('click', () => {
            if (headerRefreshButton) {
                // Trigger the header refresh button click
                headerRefreshButton.click();
            }
        });
    }
    
    // Also sync when the model list is updated
    const modelSelectObserver = new MutationObserver(() => {
        syncModelOptions();
    });
    
    modelSelectObserver.observe(headerModelSelect, { childList: true, subtree: true });
}

// Call this function after the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeSidebarModelSelector();
    handleInitialLoading();
    
    // Initialize quota exceeded modal close button
    const closeQuotaModal = document.getElementById('close-quota-modal');
    const confirmQuotaModal = document.getElementById('confirm-quota-modal');
    const quotaExceededModal = document.getElementById('quota-exceeded-modal');
    
    if (closeQuotaModal && quotaExceededModal) {
        closeQuotaModal.addEventListener('click', function() {
            quotaExceededModal.style.display = 'none';
        });
    }
    
    if (confirmQuotaModal && quotaExceededModal) {
        confirmQuotaModal.addEventListener('click', function() {
            quotaExceededModal.style.display = 'none';
        });
    }
});

// Control initial loading screen
function handleInitialLoading() {
    const loadingScreen = document.getElementById('initial-loading-screen');
    
    // Check if key elements are loaded
    function checkIfAppIsReady() {
        // Different minimum loading times for desktop vs mobile
        const isMobile = window.innerWidth <= 768;
        const minLoadTime = isMobile ? 3000 : 1500; // 3 seconds for mobile, 1.5 for desktop
        
        const startTime = Date.now();
        
        // Elements to check if they're loaded/rendered
        const elementsToCheck = [
            document.querySelector('.sidebar'),
            document.querySelector('.chat-panel'),
            document.getElementById('model-select')
        ];
        
        // Wait for minimum load time and check if elements are rendered
        setTimeout(() => {
            const allElementsLoaded = elementsToCheck.every(el => el !== null);
            
            if (allElementsLoaded) {
                // Add loaded class to trigger fade-out animation
                loadingScreen.classList.add('loaded');
                
                // Remove from DOM after animation completes
                setTimeout(() => {
                    loadingScreen.style.display = 'none';
                }, 500); // Match transition duration from CSS
            } else {
                // If not all elements are loaded, check again in 300ms
                setTimeout(checkIfAppIsReady, 300);
            }
        }, minLoadTime);
    }
    
    // Start checking if app is ready
    checkIfAppIsReady();
}
