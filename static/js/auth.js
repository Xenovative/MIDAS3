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
    // We're now using the sidebar user section instead of the floating user menu
    const sidebarUserInfo = document.getElementById('sidebar-user-info');
    const sidebarUserSection = document.querySelector('.sidebar-user-section');
    
    if (authState.logged_in) {
        // Update sidebar user info
        if (sidebarUserInfo) {
            sidebarUserInfo.textContent = `${authState.display_name || authState.username}${authState.role ? ` (${authState.role})` : ''}`;
        }
        if (sidebarUserSection) {
            sidebarUserSection.style.display = 'block';
        }
        
        hideAuthModal();
        document.body.classList.remove('auth-locked');
    } else {
        // Hide sidebar user section when logged out
        if (sidebarUserSection) {
            sidebarUserSection.style.display = 'none';
        }
        
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

// Function to update user info displays
function updateUserInfo() {
    const username = localStorage.getItem('username');
    if (username) {
        // Update user info in header
        const userInfoElement = document.getElementById('user-info');
        if (userInfoElement) {
            userInfoElement.textContent = username;
        }
        
        // Update user info in sidebar
        const sidebarUserInfo = document.getElementById('sidebar-user-info');
        if (sidebarUserInfo) {
            sidebarUserInfo.textContent = username;
        }
    }
}

// Function to handle logout
async function handleLogout() {
    try {
        // Call the server logout endpoint
        await fetch('/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
    } catch (error) {
        console.error('Error during logout:', error);
    } finally {
        // Clear local storage
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        localStorage.removeItem('userId');
        
        // Update auth state
        authState = { logged_in: false, username: null, role: null };
        
        // Update UI
        updateAuthUI();
    }
}

// Add event listener for logout button in header
const logoutButton = document.getElementById('logout-button');
if (logoutButton) {
    logoutButton.addEventListener('click', handleLogout);
}

// Add event listener for logout button in sidebar
const sidebarLogoutBtn = document.getElementById('sidebar-logout-btn');
if (sidebarLogoutBtn) {
    sidebarLogoutBtn.addEventListener('click', handleLogout);
}

// Initialize authentication
document.addEventListener('DOMContentLoaded', async function() {
    console.log('DOM loaded, initializing auth...');
    
    // Fetch auth state immediately
    await fetchAuthState();
    
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
    
    // Initialize quota management
    initQuotaManagement();
    
    // When the user tab is shown, populate the display name field
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            console.log(`Tab clicked: ${tabName}, auth state:`, authState);
            
            if (tabName === 'user' && authState.logged_in) {
                // Populate display name field with current value
                displayNameInput.value = authState.display_name || authState.username;
            } else if (tabName === 'admin' && authState.logged_in && authState.role === 'admin') {
                console.log('Admin tab clicked, loading user list...');
                // Load user list when admin tab is clicked
                loadUserList();
            }
        });
    });
    
    // Function to load the user list for admin
    async function loadUserList() {
        console.log('loadUserList called, auth state:', authState);
        
        if (!authState.logged_in || authState.role !== 'admin') {
            console.log('Not logged in as admin, skipping user list load');
            return;
        }
        
        // Show loading, hide list and error
        if (userManagementLoading) userManagementLoading.style.display = 'flex';
        if (userListContainer) userListContainer.style.display = 'none';
        if (userManagementError) userManagementError.style.display = 'none';
        
        try {
            console.log('Fetching users from API...');
            const response = await fetch('/api/users');
            console.log('API response status:', response.status);
            const data = await response.json();
            console.log('API response data:', data);
            
            if (data.status === 'success' && data.users) {
                // Clear existing list
                if (userList) userList.innerHTML = '';
                
                // Add users to the list
                if (data.users.length === 0) {
                    console.log('No users returned from API');
                    if (userManagementError) {
                        userManagementError.textContent = 'No users found in the system.';
                        userManagementError.style.display = 'block';
                    }
                    if (userManagementLoading) userManagementLoading.style.display = 'none';
                } else {
                    console.log(`Adding ${data.users.length} users to the list`);
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
                    
                    // Show the user list container after populating
                    if (userListContainer) userListContainer.style.display = 'block';
                    if (userManagementLoading) userManagementLoading.style.display = 'none';
                }
            } else {
                console.error('Failed to get users:', data.message || 'Unknown error');
                if (userManagementError) {
                    userManagementError.textContent = data.message || 'Failed to load users';
                    userManagementError.style.display = 'block';
                }
                if (userManagementLoading) userManagementLoading.style.display = 'none';
            }
        } catch (error) {
            console.error('Error loading user list:', error);
            if (userManagementError) {
                userManagementError.textContent = 'Error loading users: ' + (error.message || 'Unknown error');
                userManagementError.style.display = 'block';
            }
            if (userManagementLoading) userManagementLoading.style.display = 'none';
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
        authOverlay.addEventListener('click', function(event) {
            // Only close if the user clicked directly on the overlay
            // and not on any of its children
            if (event.target === authOverlay && confirm("Are you sure you want to close this dialog?")) {
                hideAuthModal();
            }
        });
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
        // --- Registration input validation and password strength checker ---
        const registerUsernameInput = document.getElementById('register-username');
        const registerPasswordInput = document.getElementById('register-password');
        const registerError = document.getElementById('register-error');
        const passwordInfoIcon = document.getElementById('password-info-icon');
        const passwordTooltip = document.getElementById('password-tooltip');
        const strengthLabel = document.getElementById('strength-label');
        const strengthRequirements = document.getElementById('strength-requirements');
        
        // Show/hide password requirements tooltip
        if (passwordInfoIcon && passwordTooltip) {
            passwordInfoIcon.addEventListener('mouseenter', function() {
                passwordTooltip.style.display = 'block';
            });
            
            passwordInfoIcon.addEventListener('mouseleave', function() {
                passwordTooltip.style.display = 'none';
            });
            
            // Also allow clicking to toggle
            passwordInfoIcon.addEventListener('click', function() {
                if (passwordTooltip.style.display === 'none') {
                    passwordTooltip.style.display = 'block';
                } else {
                    passwordTooltip.style.display = 'none';
                }
            });
        }
        
        function checkPasswordStrength(password) {
            // Initialize criteria checks
            const criteria = {
                length: password.length >= 8,
                uppercase: /[A-Z]/.test(password),
                lowercase: /[a-z]/.test(password),
                numbers: /[0-9]/.test(password),
                special: /[^A-Za-z0-9]/.test(password)
            };
            
            // Count how many criteria are met
            const score = Object.values(criteria).filter(Boolean).length;
            
            return { score, criteria };
        }
        
        function updatePasswordStrength(password) {
            const fill = document.getElementById('password-strength-fill');
            if (!fill) return;
            
            const { score, criteria } = checkPasswordStrength(password);
            
            // Update the strength bar
            let strength = '', color = '';
            switch (score) {
                case 0:
                case 1:
                    strength = 'Very Weak'; color = '#e57373'; break;
                case 2:
                    strength = 'Weak'; color = '#ffb74d'; break;
                case 3:
                    strength = 'Moderate'; color = '#ffd54f'; break;
                case 4:
                    strength = 'Strong'; color = '#81c784'; break;
                case 5:
                    strength = 'Very Strong'; color = '#388e3c'; break;
            }
            
            // Update UI elements
            fill.style.width = (score * 20) + '%';
            fill.style.backgroundColor = color;
            
            if (strengthLabel) {
                strengthLabel.textContent = strength;
                strengthLabel.style.color = color;
            }
            
            // Show missing requirements
            if (strengthRequirements) {
                const missing = [];
                if (!criteria.length) missing.push('8+ chars');
                if (!criteria.uppercase) missing.push('uppercase');
                if (!criteria.lowercase) missing.push('lowercase');
                if (!criteria.numbers) missing.push('number');
                if (!criteria.special) missing.push('symbol');
                
                if (missing.length > 0) {
                    strengthRequirements.textContent = 'Missing: ' + missing.join(', ');
                } else {
                    strengthRequirements.textContent = 'All requirements met!';
                    strengthRequirements.style.color = '#388e3c';
                }
            }
            
            return score;
        }
        
        if (registerPasswordInput) {
            registerPasswordInput.addEventListener('input', function() {
                updatePasswordStrength(this.value);
            });
            
            // Initialize with empty password
            updatePasswordStrength('');
        }
        
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const username = registerUsernameInput.value.trim();
            const password = registerPasswordInput.value;
            // --- Username validation ---
            if (username.length < 3 || username.length > 32) {
                registerError.textContent = 'Username must be between 3 and 32 characters.';
                return;
            }
            if (!/^[A-Za-z0-9_]+$/.test(username)) {
                registerError.textContent = 'Username can only contain letters, numbers, and underscores.';
                return;
            }
            // --- Password validation ---
            if (password.length < 8) {
                registerError.textContent = 'Password must be at least 8 characters.';
                return;
            }
            const { score } = checkPasswordStrength(password);
            if (score < 3) {
                registerError.textContent = 'Password is too weak. Use uppercase, lowercase, numbers, and symbols.';
                return;
            }
            // --- Proceed with registration ---
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
                    registerError.textContent = data.message || 'Registration failed';
                }
            } catch (err) {
                registerError.textContent = 'Server error';
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
    
    // User menu auto-hide functionality
    const userMenu = document.getElementById('user-menu');
    const userInfo = document.getElementById('user-info');
    
    if (userMenu && userInfo) {
        // Function to check if menu is visible
        function isMenuVisible() {
            // Check computed style instead of just the inline style
            const computedStyle = window.getComputedStyle(userMenu);
            return computedStyle.display !== 'none';
        }
        
        // Function to toggle menu visibility
        function toggleUserMenu() {
            if (isMenuVisible()) {
                userMenu.style.display = 'none';
            } else {
                userMenu.style.display = 'flex';
            }
        }
        
        // Add click event listener to the document
        document.addEventListener('click', function(e) {
            // If the click is outside the user menu and the menu is visible
            if (isMenuVisible() && 
                !userMenu.contains(e.target) && 
                !e.target.matches('#user-info, #user-info *')) {
                userMenu.style.display = 'none';
            }
        });
        
        // Toggle menu when user info is clicked
        userInfo.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent document click from immediately closing
            toggleUserMenu();
        });
        
        // Auto-hide after clicking an option inside the menu
        userMenu.addEventListener('click', function(e) {
            // If the click is on a button or link inside the menu
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') {
                // Small delay to allow the click to register before hiding
                setTimeout(function() {
                    userMenu.style.display = 'none';
                }, 100);
            }
        });
        
        // For mobile: add touch events to improve responsiveness
        if ('ontouchstart' in window) {
            document.addEventListener('touchstart', function(e) {
                if (isMenuVisible() && 
                    !userMenu.contains(e.target) && 
                    !e.target.matches('#user-info, #user-info *')) {
                    userMenu.style.display = 'none';
                }
            });
        }
    }
    
    // Check if we should show the admin tab by default
    if (authState.logged_in && authState.role === 'admin') {
        console.log('User is admin, initializing admin panel...');
        
        // Initialize admin tab elements
        const adminTabButton = document.querySelector('.tab-button[data-tab="admin"]');
        if (adminTabButton) {
            console.log('Admin tab button found, ensuring it is visible');
            adminTabButton.style.display = '';
            
            // If admin tab is currently active, load the user list
            if (adminTabButton.classList.contains('active')) {
                console.log('Admin tab is active, loading user list');
                setTimeout(() => {
                    loadUserList();
                    loadUsersForQuotaManagement();
                }, 500); // Small delay to ensure DOM is ready
            }
        }
    }
});

// Function to handle login success
function handleLoginSuccess(username, token, userId) {
    localStorage.setItem('token', token);
    localStorage.setItem('username', username);
    localStorage.setItem('userId', userId);
    
    // Close the auth modal
    closeAuthModal();
    
    // Update the UI to show the user is logged in
    document.body.classList.add('logged-in');
    
    // Update user info displays
    updateUserInfo();
    
    // Reload conversations if needed
    if (typeof loadConversations === 'function') {
        loadConversations();
    }
}

// --- User Quota Management Functions ---

// Function to load users for quota management
function loadUsersForQuotaManagement() {
    const quotaUserSelect = document.getElementById('quota-user-select');
    if (!quotaUserSelect) return;
    
    // Clear existing options except the default one
    while (quotaUserSelect.options.length > 1) {
        quotaUserSelect.remove(1);
    }
    
    // Only proceed if user is admin
    if (!authState.logged_in || authState.role !== 'admin') return;
    
    console.log('Loading users for quota management...');
    
    // Fetch users
    fetch('/api/users')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.users) {
                console.log(`Got ${data.users.length} users for quota management`);
                
                // Add users to dropdown
                data.users.forEach(user => {
                    // Check if user already exists in dropdown to prevent duplicates
                    const exists = Array.from(quotaUserSelect.options).some(option => 
                        option.value === user.id.toString());
                    
                    if (!exists) {
                        const option = document.createElement('option');
                        option.value = user.id;
                        option.textContent = user.username;
                        quotaUserSelect.appendChild(option);
                    }
                });
            } else {
                console.error('Failed to load users for quota management');
            }
        })
        .catch(error => {
            console.error('Error loading users for quota management:', error);
        });
}

// Function to load quota settings for a user
function loadUserQuota(userId) {
    if (!userId) return;
    
    const quotaSettings = document.getElementById('quota-settings');
    const quotaUpdateStatus = document.getElementById('quota-update-status');
    
    // Show loading status
    if (quotaUpdateStatus) {
        quotaUpdateStatus.textContent = 'Loading quota settings...';
        quotaUpdateStatus.className = 'update-status';
    }
    
    console.log(`Loading quota settings for user ID: ${userId}`);
    
    // Fetch quota settings
    fetch(`/api/users/${userId}/quota`)
        .then(response => response.json())
        .then(data => {
            console.log('Received quota data:', data);
            if (data.status === 'success' && data.quota) {
                const quota = data.quota;
                console.log('Quota settings loaded:', quota);
                
                // Populate form fields - use explicit null checks to handle 0 values correctly
                const dailyLimitInput = document.getElementById('daily-message-limit');
                const monthlyLimitInput = document.getElementById('monthly-message-limit');
                const maxAttachmentSizeInput = document.getElementById('max-attachment-size');
                
                dailyLimitInput.value = quota.daily_message_limit !== null ? quota.daily_message_limit : '';
                monthlyLimitInput.value = quota.monthly_message_limit !== null ? quota.monthly_message_limit : '';
                maxAttachmentSizeInput.value = quota.max_attachment_size_kb !== null ? quota.max_attachment_size_kb : '';
                
                console.log('Updated form fields:', {
                    daily: dailyLimitInput.value,
                    monthly: monthlyLimitInput.value,
                    attachment: maxAttachmentSizeInput.value
                });
                
                // Update usage stats
                document.getElementById('daily-usage').textContent = quota.messages_used_today || '0';
                document.getElementById('daily-limit').textContent = quota.daily_message_limit !== null ? quota.daily_message_limit : '0';
                document.getElementById('monthly-usage').textContent = quota.messages_used_month || '0';
                document.getElementById('monthly-limit').textContent = quota.monthly_message_limit !== null ? quota.monthly_message_limit : '0';
                
                // Show quota settings
                quotaSettings.style.display = 'block';
                
                // Clear status
                if (quotaUpdateStatus) {
                    quotaUpdateStatus.textContent = '';
                }
            } else {
                console.error('Failed to load quota settings:', data.message);
                if (quotaUpdateStatus) {
                    quotaUpdateStatus.textContent = data.message || 'Failed to load quota settings';
                    quotaUpdateStatus.className = 'update-status error';
                }
            }
        })
        .catch(error => {
            console.error('Error loading quota settings:', error);
            if (quotaUpdateStatus) {
                quotaUpdateStatus.textContent = 'Error loading quota settings';
                quotaUpdateStatus.className = 'update-status error';
            }
        });
}

// Function to save quota settings
function saveUserQuota(userId) {
    if (!userId) {
        console.error('Cannot save quota: No user ID provided');
        return;
    }
    
    console.log('Saving quota settings for user ID:', userId);
    const quotaUpdateStatus = document.getElementById('quota-update-status');
    
    // Get values from form
    const dailyLimitInput = document.getElementById('daily-message-limit');
    const monthlyLimitInput = document.getElementById('monthly-message-limit');
    const maxAttachmentSizeInput = document.getElementById('max-attachment-size');
    
    // Parse values, handling empty strings properly
    const dailyLimit = dailyLimitInput.value.trim() === '' ? null : parseInt(dailyLimitInput.value);
    const monthlyLimit = monthlyLimitInput.value.trim() === '' ? null : parseInt(monthlyLimitInput.value);
    const maxAttachmentSize = maxAttachmentSizeInput.value.trim() === '' ? null : parseInt(maxAttachmentSizeInput.value);
    
    console.log('Quota values to save:', {
        dailyLimit,
        monthlyLimit,
        maxAttachmentSize
    });
    
    // Validate input
    if (dailyLimit !== null && dailyLimit < 0) {
        quotaUpdateStatus.textContent = 'Daily limit must be a positive number';
        quotaUpdateStatus.className = 'update-status error';
        return;
    }
    
    if (monthlyLimit !== null && monthlyLimit < 0) {
        quotaUpdateStatus.textContent = 'Monthly limit must be a positive number';
        quotaUpdateStatus.className = 'update-status error';
        return;
    }
    
    if (maxAttachmentSize !== null && maxAttachmentSize < 0) {
        quotaUpdateStatus.textContent = 'Max attachment size must be a positive number';
        quotaUpdateStatus.className = 'update-status error';
        return;
    }
    
    // Show saving status
    quotaUpdateStatus.textContent = 'Saving quota settings...';
    quotaUpdateStatus.className = 'update-status';
    
    const requestData = {
        daily_message_limit: dailyLimit,
        monthly_message_limit: monthlyLimit,
        max_attachment_size_kb: maxAttachmentSize
    };
    
    console.log('Sending quota update request:', {
        url: `/api/users/${userId}/quota`,
        method: 'PUT',
        data: requestData
    });
    
    // Send update request
    fetch(`/api/users/${userId}/quota`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        console.log('Quota update response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('Quota update response data:', data);
        if (data.status === 'success') {
            quotaUpdateStatus.textContent = 'Quota settings saved successfully';
            quotaUpdateStatus.className = 'update-status success';
            
            // Update the input fields with the saved values to ensure they persist
            if (data.quota) {
                dailyLimitInput.value = data.quota.daily_message_limit !== null ? data.quota.daily_message_limit : '';
                monthlyLimitInput.value = data.quota.monthly_message_limit !== null ? data.quota.monthly_message_limit : '';
                maxAttachmentSizeInput.value = data.quota.max_attachment_size_kb !== null ? data.quota.max_attachment_size_kb : '';
                
                // Update usage stats
                document.getElementById('daily-limit').textContent = data.quota.daily_message_limit !== null ? data.quota.daily_message_limit : '0';
                document.getElementById('monthly-limit').textContent = data.quota.monthly_message_limit !== null ? data.quota.monthly_message_limit : '0';
            } else {
                // If no quota data returned, reload to get current values
                setTimeout(() => loadUserQuota(userId), 500);
            }
        } else {
            quotaUpdateStatus.textContent = data.message || 'Failed to save quota settings';
            quotaUpdateStatus.className = 'update-status error';
        }
    })
    .catch(error => {
        console.error('Error saving quota settings:', error);
        quotaUpdateStatus.textContent = 'Error saving quota settings';
        quotaUpdateStatus.className = 'update-status error';
    });
}

// Initialize quota management
function initQuotaManagement() {
    console.log('Initializing quota management');
    const quotaUserSelect = document.getElementById('quota-user-select');
    const saveQuotaBtn = document.getElementById('save-quota-btn');
    const resetQuotaBtn = document.getElementById('reset-quota-btn');
    
    console.log('Quota elements:', {
        quotaUserSelect: quotaUserSelect ? 'Found' : 'Not found',
        saveQuotaBtn: saveQuotaBtn ? 'Found' : 'Not found',
        resetQuotaBtn: resetQuotaBtn ? 'Found' : 'Not found'
    });
    
    if (quotaUserSelect) {
        // Load users when admin tab is shown
        const adminTabButton = document.querySelector('.tab-button[data-tab="admin"]');
        if (adminTabButton) {
            console.log('Adding click listener to admin tab button');
            adminTabButton.addEventListener('click', function() {
                console.log('Admin tab clicked, auth state:', authState);
                if (authState.role === 'admin') {
                    // Clear the dropdown first
                    while (quotaUserSelect.options.length > 1) {
                        quotaUserSelect.remove(1);
                    }
                    loadUsersForQuotaManagement();
                }
            });
        } else {
            console.error('Admin tab button not found');
        }
        
        // Load quota settings when user is selected
        quotaUserSelect.addEventListener('change', function() {
            const userId = this.value;
            console.log('User selected for quota management:', userId);
            if (userId) {
                loadUserQuota(userId);
            } else {
                document.getElementById('quota-settings').style.display = 'none';
            }
        });
    }
    
    if (saveQuotaBtn) {
        console.log('Adding click listener to save quota button');
        saveQuotaBtn.addEventListener('click', function() {
            console.log('Save quota button clicked');
            const userId = document.getElementById('quota-user-select').value;
            console.log('User ID for quota save:', userId);
            if (userId) {
                saveUserQuota(userId);
            } else {
                console.error('No user selected for quota save');
                const quotaUpdateStatus = document.getElementById('quota-update-status');
                if (quotaUpdateStatus) {
                    quotaUpdateStatus.textContent = 'Please select a user first';
                    quotaUpdateStatus.className = 'update-status error';
                }
            }
        });
    } else {
        console.error('Save quota button not found');
    }
    
    if (resetQuotaBtn) {
        console.log('Adding click listener to reset quota button');
        resetQuotaBtn.addEventListener('click', function() {
            console.log('Reset quota button clicked');
            const userId = document.getElementById('quota-user-select').value;
            console.log('User ID for quota reset:', userId);
            if (!userId) {
                console.error('No user selected for quota reset');
                const quotaUpdateStatus = document.getElementById('quota-update-status');
                if (quotaUpdateStatus) {
                    quotaUpdateStatus.textContent = 'Please select a user first';
                    quotaUpdateStatus.className = 'update-status error';
                }
                return;
            }
            
            const quotaUpdateStatus = document.getElementById('quota-update-status');
            
            if (confirm('Are you sure you want to reset usage counters for this user?')) {
                // Show resetting status
                quotaUpdateStatus.textContent = 'Resetting usage counters...';
                quotaUpdateStatus.className = 'update-status';
                
                // Send reset request
                fetch(`/api/users/${userId}/quota/reset`, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        quotaUpdateStatus.textContent = 'Usage counters reset successfully';
                        quotaUpdateStatus.className = 'update-status success';
                        
                        // Reload quota settings to show updated values
                        loadUserQuota(userId);
                    } else {
                        quotaUpdateStatus.textContent = data.message || 'Failed to reset usage counters';
                        quotaUpdateStatus.className = 'update-status error';
                    }
                })
                .catch(error => {
                    console.error('Error resetting usage counters:', error);
                    quotaUpdateStatus.textContent = 'Error resetting usage counters';
                    quotaUpdateStatus.className = 'update-status error';
                });
            }
        });
    } else {
        console.error('Reset quota button not found');
    }
}

// Call initQuotaManagement when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initQuotaManagement();
});
