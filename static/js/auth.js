// Authentication state
let authState = { logged_in: false, username: null, role: null };

// --- Authentication Functions ---
async function fetchAuthState() {
    try {
        const res = await fetch('/me');
        const data = await res.json();
        authState = data;
        updateAuthUI();
        return data;
    } catch (e) {
        console.error("Error fetching auth state:", e);
        authState = { logged_in: false };
        updateAuthUI();
        return { logged_in: false };
    }
}

function updateAuthUI() {
    const userMenu = document.getElementById('user-menu');
    const userInfo = document.getElementById('user-info');
    
    if (authState.logged_in) {
        if (userMenu) userMenu.style.display = 'flex';
        if (userInfo) userInfo.textContent = `${authState.display_name || authState.username} (${authState.role})`;
        hideAuthModal();
        document.body.classList.remove('auth-locked');
    } else {
        if (userMenu) userMenu.style.display = 'none';
        showAuthModal();
        document.body.classList.add('auth-locked');
    }
    
    // Hide admin-only UI if not admin
    if (authState.role !== 'admin') {
        document.querySelectorAll('.admin-only').forEach(el => el.style.display = 'none');
    } else {
        document.querySelectorAll('.admin-only').forEach(el => el.style.display = '');
    }
    
    lockChatUIIfNeeded();
}

function showAuthModal() {
    const authOverlay = document.getElementById('auth-overlay');
    const authModal = document.getElementById('auth-modal');
    
    // Hide any other modals that might be open
    document.querySelectorAll('.modal:not(#auth-modal)').forEach(modal => {
        modal.style.display = 'none';
    });
    
    if (authOverlay && authModal) {
        authOverlay.style.display = 'block';
        authModal.style.display = 'flex';
    }
}

function hideAuthModal() {
    const authOverlay = document.getElementById('auth-overlay');
    const authModal = document.getElementById('auth-modal');
    const loginError = document.getElementById('login-error');
    const registerError = document.getElementById('register-error');
    
    if (authOverlay) authOverlay.style.display = 'none';
    if (authModal) authModal.style.display = 'none';
    if (loginError) loginError.textContent = '';
    if (registerError) registerError.textContent = '';
}

function lockChatUIIfNeeded() {
    const mainContent = document.querySelector('.main-content');
    if (!authState.logged_in) {
        if (mainContent) {
            mainContent.style.opacity = '0.5';
            mainContent.style.pointerEvents = 'none';
        }
    } else {
        if (mainContent) {
            mainContent.style.opacity = '1';
            mainContent.style.pointerEvents = 'auto';
        }
    }
}

// Initialize authentication
document.addEventListener('DOMContentLoaded', function() {
    // Fetch auth state immediately
    fetchAuthState();
    
    // Set up auth UI event listeners
    const showLogin = document.getElementById('show-login');
    const showRegister = document.getElementById('show-register');
    const closeAuthModal = document.getElementById('close-auth-modal');
    const authOverlay = document.getElementById('auth-overlay');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const logoutBtn = document.getElementById('logout-btn');
    
    // User profile update elements
    const displayNameInput = document.getElementById('display-name');
    const currentPasswordInput = document.getElementById('current-password');
    const newPasswordInput = document.getElementById('new-password');
    const confirmPasswordInput = document.getElementById('confirm-password');
    const savePreferencesBtn = document.getElementById('save-preferences');
    const passwordUpdateStatus = document.getElementById('password-update-status');
    
    // Admin tab elements
    const userManagementLoading = document.getElementById('user-management-loading');
    const userListContainer = document.getElementById('user-list-container');
    const userList = document.getElementById('user-list');
    const userManagementError = document.getElementById('user-management-error');
    
    // When the user tab is shown, populate the display name field
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            if (tabName === 'user' && authState.logged_in) {
                // Populate display name field with current value
                displayNameInput.value = authState.display_name || authState.username;
            } else if (tabName === 'admin' && authState.logged_in && authState.role === 'admin') {
                // Load user list when admin tab is clicked
                loadUserList();
            }
        });
    });
    
    // Function to load the user list for admin
    async function loadUserList() {
        if (!authState.logged_in || authState.role !== 'admin') return;
        
        // Show loading, hide list and error
        if (userManagementLoading) userManagementLoading.style.display = 'flex';
        if (userListContainer) userListContainer.style.display = 'none';
        if (userManagementError) userManagementError.style.display = 'none';
        
        try {
            const response = await fetch('/api/users');
            const data = await response.json();
            
            if (data.status === 'success' && data.users) {
                // Clear existing list
                if (userList) userList.innerHTML = '';
                
                // Add users to the list
                data.users.forEach(user => {
                    const row = document.createElement('tr');
                    
                    // Username cell
                    const usernameCell = document.createElement('td');
                    usernameCell.textContent = user.username;
                    row.appendChild(usernameCell);
                    
                    // Display name cell
                    const displayNameCell = document.createElement('td');
                    displayNameCell.textContent = user.display_name || user.username;
                    row.appendChild(displayNameCell);
                    
                    // Role cell with dropdown for editing
                    const roleCell = document.createElement('td');
                    const roleSelect = document.createElement('select');
                    roleSelect.className = 'user-role-select';
                    roleSelect.disabled = user.id === authState.id; // Can't change your own role
                    
                    const userOption = document.createElement('option');
                    userOption.value = 'user';
                    userOption.textContent = 'User';
                    userOption.selected = user.role === 'user';
                    roleSelect.appendChild(userOption);
                    
                    const adminOption = document.createElement('option');
                    adminOption.value = 'admin';
                    adminOption.textContent = 'Admin';
                    adminOption.selected = user.role === 'admin';
                    roleSelect.appendChild(adminOption);
                    
                    // Add change event to update role
                    roleSelect.addEventListener('change', async function() {
                        const newRole = this.value;
                        try {
                            const response = await fetch(`/api/users/${user.id}`, {
                                method: 'PUT',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ role: newRole })
                            });
                            
                            const result = await response.json();
                            if (result.status === 'success') {
                                if (window.showNotification) {
                                    window.showNotification(`Role updated for ${user.username}`, 'success');
                                }
                            } else {
                                if (window.showNotification) {
                                    window.showNotification(result.message || 'Failed to update role', 'error');
                                }
                                // Reset to original value
                                this.value = user.role;
                            }
                        } catch (error) {
                            console.error('Error updating user role:', error);
                            if (window.showNotification) {
                                window.showNotification('Error updating role', 'error');
                            }
                            // Reset to original value
                            this.value = user.role;
                        }
                    });
                    
                    roleCell.appendChild(roleSelect);
                    row.appendChild(roleCell);
                    
                    // Actions cell
                    const actionsCell = document.createElement('td');
                    
                    // Delete button (disabled for your own account)
                    if (user.id !== authState.id) {
                        const deleteButton = document.createElement('button');
                        deleteButton.className = 'user-action-button delete';
                        deleteButton.textContent = 'Delete';
                        deleteButton.addEventListener('click', async function() {
                            if (confirm(`Are you sure you want to delete user ${user.username}? This cannot be undone.`)) {
                                try {
                                    const response = await fetch(`/api/users/${user.id}`, {
                                        method: 'DELETE'
                                    });
                                    
                                    const result = await response.json();
                                    if (result.status === 'success') {
                                        if (window.showNotification) {
                                            window.showNotification(`User ${user.username} deleted`, 'success');
                                        }
                                        // Remove row from table
                                        row.remove();
                                    } else {
                                        if (window.showNotification) {
                                            window.showNotification(result.message || 'Failed to delete user', 'error');
                                        }
                                    }
                                } catch (error) {
                                    console.error('Error deleting user:', error);
                                    if (window.showNotification) {
                                        window.showNotification('Error deleting user', 'error');
                                    }
                                }
                            }
                        });
                        actionsCell.appendChild(deleteButton);
                    } else {
                        const selfLabel = document.createElement('span');
                        selfLabel.textContent = '(You)';
                        selfLabel.style.fontStyle = 'italic';
                        selfLabel.style.color = 'var(--text-secondary)';
                        actionsCell.appendChild(selfLabel);
                    }
                    
                    // Add change password button (admin only, not for self)
                    if (authState.role === 'admin' && user.id !== authState.id) {
                        const changePwdButton = document.createElement('button');
                        changePwdButton.textContent = 'Change Password';
                        changePwdButton.className = 'user-action-button'; // Match existing button styling
                        changePwdButton.addEventListener('click', function() {
                            // Check if password form already exists and remove it if it does
                            const existingForm = document.getElementById(`pwd-form-${user.id}`);
                            if (existingForm) {
                                existingForm.remove();
                                return; // Toggle off if already showing
                            }
                            
                            // Create inline password form
                            const pwdForm = document.createElement('div');
                            pwdForm.id = `pwd-form-${user.id}`;
                            pwdForm.className = 'inline-password-form';
                            pwdForm.innerHTML = `
                                <div class="form-row">
                                    <input type="password" id="new-pwd-${user.id}" 
                                           class="preference-input" placeholder="New password" 
                                           minlength="6" required>
                                    <button type="button" class="primary-button save-pwd-btn">Save</button>
                                    <button type="button" class="secondary-button cancel-pwd-btn">Cancel</button>
                                </div>
                                <div class="pwd-status" id="pwd-status-${user.id}"></div>
                            `;
                            
                            // Insert after the current row
                            row.parentNode.insertBefore(pwdForm, row.nextSibling);
                            
                            // Focus the input
                            document.getElementById(`new-pwd-${user.id}`).focus();
                            
                            // Add event listeners to the buttons
                            pwdForm.querySelector('.save-pwd-btn').addEventListener('click', async function() {
                                const newPwd = document.getElementById(`new-pwd-${user.id}`).value;
                                const statusEl = document.getElementById(`pwd-status-${user.id}`);
                                
                                if (!newPwd || newPwd.length < 6) {
                                    statusEl.textContent = 'Password must be at least 6 characters';
                                    statusEl.className = 'pwd-status error';
                                    return;
                                }
                                
                                try {
                                    const response = await fetch(`/api/users/${user.id}/password`, {
                                        method: 'PUT',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ new_password: newPwd })
                                    });
                                    
                                    const data = await response.json();
                                    if (data.status === 'success') {
                                        statusEl.textContent = 'Password updated successfully';
                                        statusEl.className = 'pwd-status success';
                                        
                                        // Remove the form after 2 seconds
                                        setTimeout(() => {
                                            pwdForm.remove();
                                        }, 2000);
                                    } else {
                                        statusEl.textContent = data.message || 'Failed to update password';
                                        statusEl.className = 'pwd-status error';
                                    }
                                } catch (error) {
                                    statusEl.textContent = 'Server error updating password';
                                    statusEl.className = 'pwd-status error';
                                }
                            });
                            
                            // Cancel button closes the form
                            pwdForm.querySelector('.cancel-pwd-btn').addEventListener('click', function() {
                                pwdForm.remove();
                            });
                        });
                        
                        actionsCell.appendChild(changePwdButton);
                    }
                    
                    row.appendChild(actionsCell);
                    userList.appendChild(row);
                });
                
                // Show list, hide loading and error
                if (userManagementLoading) userManagementLoading.style.display = 'none';
                if (userListContainer) userListContainer.style.display = 'block';
            } else {
                throw new Error(data.message || 'Failed to load users');
            }
        } catch (error) {
            console.error('Error loading users:', error);
            
            // Show error, hide loading and list
            if (userManagementLoading) userManagementLoading.style.display = 'none';
            if (userListContainer) userListContainer.style.display = 'none';
            if (userManagementError) {
                userManagementError.textContent = 'Failed to load users. Please try again.';
                userManagementError.style.display = 'block';
            }
        }
    }
    
    // Handle saving user profile updates
    if (savePreferencesBtn) {
        const originalClickHandler = savePreferencesBtn.onclick;
        
        savePreferencesBtn.onclick = async function(e) {
            // Get the active tab
            const activeTab = document.querySelector('.tab-button.active').getAttribute('data-tab');
            
            // If we're on the user tab, handle profile updates
            if (activeTab === 'user') {
                e.preventDefault();
                
                // Clear previous status messages
                if (passwordUpdateStatus) {
                    passwordUpdateStatus.textContent = '';
                    passwordUpdateStatus.className = 'update-status';
                }
                
                const updateData = {};
                let hasPasswordUpdate = false;
                
                // Check if we're updating the display name
                if (displayNameInput && displayNameInput.value.trim()) {
                    updateData.display_name = displayNameInput.value.trim();
                }
                
                // Check if we're updating the password
                if (currentPasswordInput && currentPasswordInput.value &&
                    newPasswordInput && newPasswordInput.value) {
                    
                    // Validate password confirmation
                    if (newPasswordInput.value !== confirmPasswordInput.value) {
                        passwordUpdateStatus.textContent = 'New passwords do not match';
                        passwordUpdateStatus.className = 'update-status error';
                        return;
                    }
                    
                    updateData.current_password = currentPasswordInput.value;
                    updateData.new_password = newPasswordInput.value;
                    hasPasswordUpdate = true;
                }
                
                // Only proceed if we have something to update
                if (Object.keys(updateData).length > 0) {
                    try {
                        const response = await fetch('/update_profile', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(updateData)
                        });
                        
                        const result = await response.json();
                        
                        if (result.status === 'success') {
                            // Show success message
                            passwordUpdateStatus.textContent = 'Profile updated successfully';
                            passwordUpdateStatus.className = 'update-status success';
                            
                            // Clear password fields
                            if (hasPasswordUpdate) {
                                currentPasswordInput.value = '';
                                newPasswordInput.value = '';
                                confirmPasswordInput.value = '';
                            }
                            
                            // Refresh auth state to get updated display name
                            await fetchAuthState();
                        } else {
                            // Show error message
                            passwordUpdateStatus.textContent = result.message || 'Failed to update profile';
                            passwordUpdateStatus.className = 'update-status error';
                        }
                    } catch (error) {
                        console.error('Error updating profile:', error);
                        passwordUpdateStatus.textContent = 'An error occurred while updating your profile';
                        passwordUpdateStatus.className = 'update-status error';
                    }
                }
                
                return;
            }
            
            // If not on the user tab, call the original handler
            if (originalClickHandler) {
                originalClickHandler.call(this, e);
            }
        };
    }
    
    if (showLogin) {
        showLogin.addEventListener('click', function() {
            document.getElementById('login-form').style.display = '';
            document.getElementById('register-form').style.display = 'none';
            this.classList.add('active');
            document.getElementById('show-register').classList.remove('active');
        });
    }
    
    if (showRegister) {
        showRegister.addEventListener('click', function() {
            document.getElementById('login-form').style.display = 'none';
            document.getElementById('register-form').style.display = '';
            this.classList.add('active');
            document.getElementById('show-login').classList.remove('active');
        });
    }
    
    if (closeAuthModal) {
        closeAuthModal.addEventListener('click', hideAuthModal);
    }
    
    if (authOverlay) {
        authOverlay.addEventListener('click', hideAuthModal);
    }
    
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const username = document.getElementById('login-username').value;
            const password = document.getElementById('login-password').value;
            const rememberMe = document.getElementById('remember-me').checked;
            try {
                const res = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    await fetchAuthState();
                    hideAuthModal();
                    if (rememberMe) {
                        localStorage.setItem('midas3_remembered_user', username);
                    } else {
                        localStorage.removeItem('midas3_remembered_user');
                    }
                    // Simpler solution: refresh the page so the landing page logic always runs
                    window.location.reload();
                    return;
                } else {
                    document.getElementById('login-error').textContent = data.message || 'Login failed';
                }
            } catch (err) {
                document.getElementById('login-error').textContent = 'Server error';
            }
        });
    }
    
    if (registerForm) {
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const username = document.getElementById('register-username').value;
            const password = document.getElementById('register-password').value;
            try {
                const res = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    await fetchAuthState();
                    hideAuthModal();
                    if (typeof window.loadConversations === 'function') {
                        window.loadConversations(true);
                    }
                    if (window.showNotification) {
                        window.showNotification('Registered successfully', 'success');
                    }
                } else {
                    document.getElementById('register-error').textContent = data.message || 'Registration failed';
                }
            } catch (err) {
                document.getElementById('register-error').textContent = 'Server error';
            }
        });
    }
    
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async function() {
            try {
                await fetch('/logout');
                
                // Clear the conversation list in the sidebar
                const conversationList = document.getElementById('conversation-list');
                if (conversationList) {
                    conversationList.innerHTML = '';
                }
                
                // Reset current conversation state
                if (window.currentConversationId) {
                    window.currentConversationId = null;
                }
                
                // Clear chat container
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    chatContainer.innerHTML = '';
                }
                
                // Reset header info
                const chatTitle = document.getElementById('chat-title');
                const chatModel = document.getElementById('chat-model');
                if (chatTitle) chatTitle.textContent = 'New Chat';
                if (chatModel) chatModel.textContent = 'Select a model';
                
                // Show landing page if it exists
                const landingPage = document.getElementById('landing-page');
                if (landingPage) {
                    landingPage.style.display = 'flex';
                    landingPage.classList.add('visible');
                }
                
                // Update auth state
                await fetchAuthState();
                
                if (window.showNotification) {
                    window.showNotification('Logged out successfully', 'info');
                }
            } catch (err) {
                if (window.showNotification) {
                    window.showNotification('Logout failed', 'error');
                }
            }
        });
    }
    
    // On DOMContentLoaded, prefill username if remembered
    const rememberedUser = localStorage.getItem('midas3_remembered_user');
    if (rememberedUser) {
        const usernameInput = document.getElementById('login-username');
        const rememberMeCheckbox = document.getElementById('remember-me');
        if (usernameInput) usernameInput.value = rememberedUser;
        if (rememberMeCheckbox) rememberMeCheckbox.checked = true;
    }
});
