import { applicationSettings, uiSettings, validationSettings } from "./config.js";

// ============================================================================
// ERROR HANDLING - Simple, reusable error utilities
// ============================================================================
export const ErrorHandler = {
  showError: (message, duration = uiSettings.errorDisplayDuration) => {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    errorDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #ff4444;
      color: white;
      padding: 12px 20px;
      border-radius: 4px;
      z-index: ${uiSettings.zIndex.error};
      box-shadow: ${uiSettings.boxShadow.error};
      font-family: 'Fredoka', sans-serif;
    `;
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
      if (errorDiv.parentNode) {
        errorDiv.parentNode.removeChild(errorDiv);
      }
    }, duration);
  },

  logError: (error, context = '') => {
    const errorMessage = context ? `${context}: ${error.message || error}` : error.message || error;
    console.error(errorMessage);
    if (error.stack) {
      console.error(error.stack);
    }
  },

  validateInput: (text) => {
    if (!text || typeof text !== 'string') {
      throw new Error('Text input is required and must be a string');
    }
    if (text.trim().length === validationSettings.textInput.minLength) {
      throw new Error('Text input cannot be empty');
    }
    if (text.length > validationSettings.textInput.maxLength) {
      throw new Error(`Text input is too long (maximum ${validationSettings.textInput.maxLength} characters)`);
    }
    return text.trim();
  },

  validateData: (data) => {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid data received from server');
    }
    if (!Array.isArray(data.surprisals)) {
      throw new Error('Invalid surprisals data received');
    }
    if (!Array.isArray(data.lengths)) {
      throw new Error('Invalid lengths data received');
    }
    if (!Array.isArray(data.frequencies_inverted)) {
      throw new Error('Invalid frequencies data received');
    }
    return data;
  },

  // Enhanced error handling with async function wrapper
  wrapAsyncFunction(fn, context) {
    return async (...args) => {
      try {
        return await fn(...args);
      } catch (error) {
        this.logError(error, context);
        this.showError(`Operation failed: ${error.message}`);
        throw error; // Re-throw for calling code to handle if needed
      }
    };
  },

  // Global error handler for uncaught errors
  setupGlobalErrorHandling() {
    window.addEventListener('error', (event) => {
      this.logError(event.error, 'Global error');
      this.showError('An unexpected error occurred');
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.logError(event.reason, 'Unhandled promise rejection');
      this.showError('A background operation failed');
    });
  },

  // Update text length limit from backend
  updateTextLimit: async () => {
    try {
      const response = await fetch('/health');
      if (response.ok) {
        const data = await response.json();
        if (data.config && data.config.max_text_length) {
          validationSettings.textInput.maxLength = data.config.max_text_length;
          applicationSettings.maxTextLength = data.config.max_text_length;
        }
      }
    } catch (error) {
      console.warn('Could not fetch text limit from backend, using default');
    }
  }
};

// ============================================================================
// VALIDATION UTILITIES - Input sanitization and validation
// ============================================================================
export const ValidationUtils = {
  sanitizeText(text) {
    return text
      .trim()
      .replace(/[<>]/g, '') // Remove potential HTML
      .substring(0, validationSettings.textInput.maxLength);  // Limit length
  },

  validateScaleSelection(scaleName) {
    // Note: This will be enhanced in the app file with actual scale data
    if (!scaleName) {
      throw new Error(`Invalid scale: ${scaleName}`);
    }
    return true;
  },

  validateModelSelection(modelName) {
    // Note: This will be enhanced in the app file with actual model data
    if (!modelName) {
      throw new Error(`Invalid model: ${modelName}`);
    }
    return true;
  },

  validateInstrumentSelection(instrumentName) {
    // Note: This will be enhanced in the app file with actual instrument data
    if (!instrumentName) {
      throw new Error(`Invalid instrument: ${instrumentName}`);
    }
    return true;
  }
};

// ============================================================================
// PERFORMANCE UTILITIES - Debouncing and throttling
// ============================================================================
export const PerformanceUtils = {
  debounce(func, wait = uiSettings.debounceDelay) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  throttle(func, limit) {
    let inThrottle;
    return function() {
      const args = arguments;
      const context = this;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }
};

// ============================================================================
// ACCESSIBILITY UTILITIES - Screen reader and keyboard navigation
// ============================================================================
export const AccessibilityUtils = {
  addKeyboardNavigation() {
    // Add tab navigation to custom buttons
    document.querySelectorAll('#scales div, #models div, #instruments div').forEach(button => {
      button.setAttribute('tabindex', '0');
      button.setAttribute('role', 'button');
      button.setAttribute('aria-label', button.textContent || button.id);
      
      // Add keyboard event handlers
      button.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          button.click();
        }
      });
    });
  },

  announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), uiSettings.announcementDelay);
  },

  updateAriaLabels() {
    // Update keyboard keys with proper labels
    document.querySelectorAll('#keyboard div').forEach(key => {
      const note = key.id;
      const isDisabled = key.classList.contains('disabled');
      key.setAttribute('aria-label', `${note} note${isDisabled ? ' (disabled)' : ''}`);
      key.setAttribute('role', 'button');
      key.setAttribute('tabindex', isDisabled ? '-1' : '0');
    });
  }
};

// ============================================================================
// UI UTILITIES - Generic UI helper functions
// ============================================================================
export const UIUtils = {
  showLoading(message = 'Processing...') {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading';
    loadingDiv.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="loading-text">${message}</div>
    `;
    loadingDiv.style.cssText = `
      position: fixed;
      top: ${uiSettings.positioning.center};
      left: ${uiSettings.positioning.center};
      transform: ${uiSettings.positioning.centerTransform};
      background: ${uiSettings.opacity.loadingBackground};
      color: white;
      padding: 20px;
      border-radius: 8px;
      z-index: ${uiSettings.zIndex.overlay};
      text-align: center;
      font-family: 'Fredoka', sans-serif;
    `;
    
    // Add spinner CSS
    const spinnerCSS = `
      .loading-spinner {
        width: ${uiSettings.spinner.size};
        height: ${uiSettings.spinner.size};
        border: ${uiSettings.spinner.borderWidth} solid #f3f3f3;
        border-top: ${uiSettings.spinner.borderWidth} solid #3498db;
        border-radius: ${uiSettings.spinner.borderRadius};
        animation: spin 1s linear infinite;
        margin: 0 auto ${uiSettings.spinner.marginBottom};
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `;
    
    if (!document.getElementById('loading-styles')) {
      const styleSheet = document.createElement('style');
      styleSheet.id = 'loading-styles';
      styleSheet.textContent = spinnerCSS;
      document.head.appendChild(styleSheet);
    }
    
    document.body.appendChild(loadingDiv);
  },

  hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
      loading.remove();
    }
  }
}; 