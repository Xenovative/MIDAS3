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
        if (userInfo) userInfo.textContent = `${authState.username} (${authState.role})`;
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
                    if (window.showNotification) {
                        window.showNotification('Logged in successfully', 'success');
                    }
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
});
