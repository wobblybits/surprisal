import { presets, scaleIntervals, notes, models, applicationSettings, audioSettings, uiSettings, validationSettings } from "./config.js";
import { ErrorHandler, ValidationUtils, PerformanceUtils, AccessibilityUtils, UIUtils } from "./utilities.js";

export class SurprisalApp {
  constructor() {
    // Import configurations
    this.presets = presets;
    this.scaleIntervals = scaleIntervals;
    this.notes = notes;
    this.models = models;
    this.appSettings = applicationSettings;
    this.audioSettings = audioSettings;
    this.uiSettings = uiSettings;

    // Application state
    this.state = {
      currentSettings: [...this.appSettings.defaultSettings],
      currentScale: null,
      currentVector: [...this.appSettings.defaultVector],
      scales: {},
      toneOutput: null,
      isLoading: false,
      
      updateSettings: (newSettings) => {
        this.state.currentSettings = newSettings;
        this.state.updateIndicator();
      },
      
      updateIndicator: () => {
        const indicator = document.getElementById('indicator');
        if (indicator) {
          // Create a copy of currentSettings and crop the model name to configured length
          const displaySettings = this.state.currentSettings.map((setting, index) => {
            if (index === this.appSettings.settingsIndices.model) {
              return setting.length > this.appSettings.modelNameDisplayLength ? 
                setting.substring(0, this.appSettings.modelNameDisplayLength) : setting;
            }
            return setting;
          });
          indicator.innerHTML = "<span>" + displaySettings.join('</span><span>') + "</span>";
        }
      },

      initScales: () => {
        const scaleGroups = Object.keys(this.scaleIntervals);
        for (const group of scaleGroups) {
          this.state.scales = {...this.state.scales, ...this.scaleIntervals[group]};
        }
        this.state.currentScale = Object.keys(this.state.scales)[0];
      },

      setLoading: (loading) => {
        this.state.isLoading = loading;
        if (loading) {
          UIUtils.showLoading();
        } else {
          UIUtils.hideLoading();
        }
      }
    };
  }

  // Audio system methods (now class methods, not nested object)
  initAudio() {
    this.synth = new Tone.Synth().toDestination();
    this.synth.volume.value = this.audioSettings.defaultVolume;
    this.sampler = new Tone.Sampler({
      urls: { "A3": "assets/audio/meow.mp3" }
    }).toDestination();
    this.state.toneOutput = this.synth;
    this.synth.set(this.presets["theremin"]);
    Tone.Transport.start(this.audioSettings.transportStart);
  }

  // Add helper method for sampler duration adjustment
  getSamplerAdjustedDuration(duration) {
    if (this.state.toneOutput === this.sampler) {
      // For sampler instruments, ensure minimum duration
      const durationInSeconds = typeof duration === 'string' ? 
        Tone.Time(duration).toSeconds() : duration;
      return Math.max(durationInSeconds, this.audioSettings.samplerMinDuration);
    }
    return duration;
  }

  async playMelody(data) {
    try {
      await Tone.start();
      const surprisals = data.surprisals;
      let pitches = [];
      let notes = [];
      
      if (this.state.currentScale === "continuous") {
        pitches = this.convertToContinuous(surprisals);
        notes = pitches;
      } else {
        pitches = this.convertToScale(surprisals);
        notes = pitches.map(d => Tone.Frequency(this.audioSettings.baseNote).transpose(d));
      }
      
      const durations = data.lengths;
      const volumes = data.frequencies_inverted;
      let delay = Tone.now();
      const offset = delay;
      const timeouts = [];

      const userInput = document.getElementById('user-input');
      if (!userInput) {
        throw new Error('User input element not found');
      }

      let startIndex = 0;
      let endIndex = 0;

      for (let i = 0; i < surprisals.length; i++) {
        timeouts.push(delay);
        endIndex = startIndex + durations[i];
        const startPos = startIndex;
        const endPos = endIndex;
        durations[i] /= this.audioSettings.volumeScaling;
        
        // Apply sampler duration adjustment
        durations[i] = this.getSamplerAdjustedDuration(durations[i]);
        
        this.state.toneOutput.volume.value = volumes[i] / this.audioSettings.volumeScaling;
        
        this.state.toneOutput.triggerAttackRelease(notes[i], durations[i], delay);  
        window.setTimeout(() => {
          try {
            const key = document.getElementById(this.convertSharpToFlat(notes[i].toNote()));
            if (key) {
              key.classList.add("highlight");
            }
            userInput.setSelectionRange(startPos, endPos);
            userInput.focus();
            window.setTimeout(() => {
              try {
                userInput.blur();
                userInput.setSelectionRange(
                  this.uiSettings.selectionReset.startPos, 
                  this.uiSettings.selectionReset.endPos
                );
                if (key) {
                  key.classList.remove("highlight");
                }
              } catch (error) {
                ErrorHandler.logError(error, 'playMelody cleanup');
              }
            }, Tone.Time(durations[i]).toSeconds() * this.audioSettings.playback.highlightDurationMultiplier);
          } catch (error) {
            ErrorHandler.logError(error, 'playMelody highlight');
          }
        }, (Tone.Time(timeouts[i]).toSeconds() - offset) * this.audioSettings.playback.timeoutConversionMs);
        delay += durations[i];
        startIndex = endIndex;
      }
    } catch (error) {
      ErrorHandler.logError(error, 'playMelody');
      ErrorHandler.showError(`Failed to play melody: ${error.message}`);
    }
  }

  convertToScale(surprisals) {
    try {
      if (!Array.isArray(surprisals)) {
        throw new Error('Surprisals must be an array');
      }
      if (!this.state.currentScale || !this.state.scales[this.state.currentScale] || !Array.isArray(this.state.scales[this.state.currentScale])) {
        throw new Error('Invalid scale configuration');
      }

      const convertedPitches = [];
      for (let i = 0; i < surprisals.length; i++) {
        if (typeof surprisals[i] !== 'number' || isNaN(surprisals[i])) {
          throw new Error('Surprisals must contain valid numbers');
        }
        const interval = Math.round(surprisals[i] / this.audioSettings.surprisalDivisor);
        const scaleLength = this.state.scales[this.state.currentScale].length;
        if (interval > scaleLength - 1) {
          convertedPitches.push(this.state.scales[this.state.currentScale][scaleLength - 1]);
        } else {
          convertedPitches.push(this.state.scales[this.state.currentScale][interval]);
        }
      }
      return convertedPitches;
    } catch (error) {
      ErrorHandler.logError(error, 'convertToScale');
      ErrorHandler.showError(`Failed to convert to scale: ${error.message}`);
      return [];
    }
  }

  convertToContinuous(surprisals) {
    try {
      if (!Array.isArray(surprisals)) {
        throw new Error('Surprisals must be an array');
      }

      const convertedPitches = [];
      const pitchCalc = this.audioSettings.continuousPitch;
      
      for (let i = 0; i < surprisals.length; i++) {
        if (typeof surprisals[i] !== 'number' || isNaN(surprisals[i])) {
          throw new Error('Surprisals must contain valid numbers');
        }
        const newPitch = Math.pow(2, (pitchCalc.baseNote + surprisals[i] / this.audioSettings.surprisalDivisor - pitchCalc.referenceNote) / pitchCalc.pitchCalculation.semitonesPerOctave) * pitchCalc.baseFrequency;
        convertedPitches.push(newPitch.toString());
      }
      // set first note to c3 since 0hz isn't audible
      convertedPitches[0] = pitchCalc.c3Frequency; 
      return convertedPitches;
    } catch (error) {
      ErrorHandler.logError(error, 'convertToContinuous');
      ErrorHandler.showError(`Failed to convert to continuous: ${error.message}`);
      return [];
    }
  }

  convertSharpToFlat(note) {
    try {
      if (!note || typeof note !== 'string') {
        throw new Error('Note must be a valid string');
      }
      if (note.indexOf("#") === validationSettings.arrayIndices.noteChar) {
        const noteIndex = this.notes.indexOf(note[0]);
        if (noteIndex === -1 || noteIndex + 1 >= this.notes.length) {
          throw new Error('Invalid note format');
        }
        return this.notes[noteIndex + 1] + note[validationSettings.arrayIndices.noteOctave];
      }
      return note;
    } catch (error) {
      ErrorHandler.logError(error, 'convertSharpToFlat');
      return note; // Return original note if conversion fails
    }
  }

  convertFlatToSymbol(note) {
    try {
      if (!note || typeof note !== 'string') {
        return note;
      }
      return note.replace("b", "&flat;");
    } catch (error) {
      ErrorHandler.logError(error, 'convertFlatToSymbol');
      return note;
    }
  }

  // API methods
  async fetchFromBackend(text) {
    try {
      const cleanText = ValidationUtils.sanitizeText(text);
      const validatedText = ErrorHandler.validateInput(cleanText);
      
      this.state.setLoading(true);
      
      // Get current model from state (convert to lowercase for API)
      const currentModel = this.state.currentSettings[this.appSettings.settingsIndices.model].toLowerCase();
      
      const response = await fetch('/process/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: validatedText,
          model: currentModel
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const validatedData = ErrorHandler.validateData(data);
      await this.playMelody(validatedData);
      
      AccessibilityUtils.announceToScreenReader('Melody composed successfully');
      
    } catch (error) {
      ErrorHandler.logError(error, 'fetchFromBackend');
      ErrorHandler.showError(`Failed to process text: ${error.message}`);
      AccessibilityUtils.announceToScreenReader('Failed to compose melody');
    } finally {
      this.state.setLoading(false);
    }
  }

  async fetchFromBackendReverse(text, scalePitch) {
    try {
      if (!this.state.currentScale || !this.state.scales[this.state.currentScale] || !Array.isArray(this.state.scales[this.state.currentScale])) {
        throw new Error('Invalid scale configuration');
      }

      const interval = this.state.scales[this.state.currentScale].indexOf(scalePitch);
      if (interval === -1) {
        throw new Error('Invalid scale pitch');
      }

      const cleanText = ValidationUtils.sanitizeText(text);
      
      // Get current model from state (convert to lowercase for API)
      const currentModel = this.state.currentSettings[this.appSettings.settingsIndices.model].toLowerCase();
      
      const response = await fetch('/reverse/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: cleanText, 
          note: interval,
          model: currentModel
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      if (!data || typeof data.input_text !== 'string' || typeof data.best_token !== 'string') {
        throw new Error('Invalid response data from server');
      }

      const userInput = document.getElementById('user-input');
      if (!userInput) {
        throw new Error('User input element not found');
      }

      const currentText = userInput.value;
      const newText = data.input_text + data.best_token;
      userInput.value = newText;
      userInput.setSelectionRange(data.input_text.length, data.input_text.length + data.best_token.length);
      userInput.focus();
      userInput.blur();
      
      AccessibilityUtils.announceToScreenReader('Text generated successfully');
      
    } catch (error) {
      ErrorHandler.logError(error, 'fetchFromBackendReverse');
      ErrorHandler.showError(`Failed to generate text: ${error.message}`);
      AccessibilityUtils.announceToScreenReader('Failed to generate text');
    }
  }

  // UI setup methods
  setupInstruments() {
    const instrumentContainer = document.getElementById("instruments");
    if (!instrumentContainer) {
      ErrorHandler.logError('Instrument container not found', 'initialization');
      return;
    }

    const instruments = Object.keys(this.presets);
    for (const instrumentName of instruments) {
      try {
        const button = document.createElement("div");
        button.id = instrumentName;
        button.innerHTML = "<img src='assets/images/instruments/" + instrumentName + ".png' />";
        button.onclick = (event) => {
          try {
            ValidationUtils.validateInstrumentSelection(instrumentName);
            
            const instrument = button.id;
            const preset = this.presets[instrument];
            if (instrument === "cat") {
              this.state.toneOutput = this.sampler;
            } else {
              this.state.toneOutput = this.synth;
              this.state.toneOutput.set(preset);
            }
            this.state.currentSettings[this.appSettings.settingsIndices.instrument] = instrument.toUpperCase();
            this.state.updateIndicator();
            
            AccessibilityUtils.announceToScreenReader(`Switched to ${instrument} instrument`);
            
          } catch (error) {
            ErrorHandler.logError(error, 'instrument selection');
            ErrorHandler.showError(`Failed to select instrument: ${error.message}`);
          }
        };
        instrumentContainer.appendChild(button);
      } catch (error) {
        ErrorHandler.logError(error, 'instrument creation');
      }
    }
  }

  setupScales() {
    const scalesContainer = document.getElementById("scales");
    if (!scalesContainer) {
      ErrorHandler.logError('Scales container not found', 'initialization');
      return;
    }

    for (const groupName in this.scaleIntervals) {
      try {
        const group = document.createElement("div");
        group.id = groupName;
        for (const scaleName in this.scaleIntervals[groupName]) {     
          const button = document.createElement("div");
          button.id = scaleName;
          button.innerHTML = scaleName.split("-")[0];
          button.onclick = (event) => {
            try {
              ValidationUtils.validateScaleSelection(scaleName);
              
              const selectedScale = button.id;
              const intervals = this.state.scales[selectedScale];
              this.state.currentScale = selectedScale;
              this.state.currentSettings[this.appSettings.settingsIndices.scale] = selectedScale.toUpperCase();
              this.state.updateIndicator();
              
              document.querySelectorAll("#keyboard div").forEach((key, index) => {
                if (this.state.scales[selectedScale].includes(index)) {
                  key.classList.remove("disabled");
                } else {
                  key.classList.add("disabled");
                }
              });
              
              AccessibilityUtils.updateAriaLabels();
              AccessibilityUtils.announceToScreenReader(`Switched to ${selectedScale} scale`);
              
            } catch (error) {
              ErrorHandler.logError(error, 'scale selection');
              ErrorHandler.showError(`Failed to select scale: ${error.message}`);
            }
          };
          group.appendChild(button);
        }
        scalesContainer.appendChild(group);
      } catch (error) {
        ErrorHandler.logError(error, 'scale creation');
      }
    }
  }

  setupModels() {
    const modelsContainer = document.getElementById("models");
    if (!modelsContainer) {
      ErrorHandler.logError('Models container not found', 'initialization');
      return;
    }

    for (const modelName of this.models) {
      try {
        const button = document.createElement("div");
        button.id = modelName;
        button.innerHTML = modelName;
        button.onclick = async (event) => {
          try {
            ValidationUtils.validateModelSelection(modelName);
            
            const selectedModel = button.id;
            this.state.currentSettings[this.appSettings.settingsIndices.model] = selectedModel.toUpperCase();
            this.state.updateIndicator();
            
            // No API call needed - model is now sent with each request
            AccessibilityUtils.announceToScreenReader(`Switched to ${selectedModel} model`);
            
          } catch (error) {
            ErrorHandler.logError(error, 'model selection');
            ErrorHandler.showError(`Failed to select model: ${error.message}`);
          }
        };
        modelsContainer.appendChild(button);
      } catch (error) {
        ErrorHandler.logError(error, 'model creation');
      }
    }
  }

  setupKeyboard() {
    const pianoContainer = document.getElementById("keyboard");
    if (!pianoContainer) {
      ErrorHandler.logError('Piano container not found', 'initialization');
      return;
    }

    const keyboardSettings = this.audioSettings.keyboard;
    const startingPitch = this.notes.indexOf(keyboardSettings.startingNote);
    let spacing = 0;
    
    for (let i = 0; i < keyboardSettings.numOctaves * keyboardSettings.semitonesPerOctave + keyboardSettings.spacing.numKeysPastOctave; i++) {
      try {
        const key = document.createElement("div");
        const note = this.notes[(i + startingPitch) % this.notes.length];
        key.id = note + (keyboardSettings.startingOctave + Math.floor(i / keyboardSettings.semitonesPerOctave));
        const scalePitch = i;
        if (note.length > validationSettings.noteLength.minLength) {
          key.style.bottom = (100 * (spacing - 0.5) / (keyboardSettings.numOctaves * keyboardSettings.spacing.whiteKeys + keyboardSettings.spacing.numKeysPastOctave)) + "%";
          key.classList.add("black-key");
        } else {
          key.style.bottom = (100 * spacing / (keyboardSettings.numOctaves * keyboardSettings.spacing.whiteKeys + keyboardSettings.spacing.numKeysPastOctave)) + "%";
          key.classList.add("white-key");
          spacing++;
        }
        key.innerHTML = this.convertFlatToSymbol(key.id);
        if (this.state.scales[this.state.currentScale] && this.state.scales[this.state.currentScale].includes(i)) {
            key.classList.remove("disabled");
        } else {
            key.classList.add("disabled");
        }
        key.onclick = (event) => {
          try {
            const adjustedDuration = this.getSamplerAdjustedDuration(this.audioSettings.noteDuration);
            this.state.toneOutput.triggerAttackRelease(key.id, adjustedDuration, Tone.now());
            const text = document.getElementById("user-input").value;
            this.fetchFromBackendReverse(text, scalePitch);
          } catch (error) {
            ErrorHandler.logError(error, 'key click');
            ErrorHandler.showError(`Failed to play key: ${error.message}`);
          }
        };
        pianoContainer.appendChild(key);
      } catch (error) {
        ErrorHandler.logError(error, 'key creation');
      }
    }
  }

  setupSubmitButton() {
    const submitButton = document.getElementById("submit");
    if (submitButton) {
      submitButton.onclick = (event) => {
        try {
          const text = document.getElementById("user-input").value;
          this.fetchFromBackend(text);
        } catch (error) {
          ErrorHandler.logError(error, 'submit button');
          ErrorHandler.showError(`Failed to submit: ${error.message}`);
        }
      };
    }
  }

  setupKeyboardEvents() {
    const lastKeyPress = {
      key: null,
      time: null,
      note: null,
    };
    
    // Debounced key handling for better performance
    const debouncedKeyHandler = PerformanceUtils.debounce((key, action) => {
      try {
        if (document.activeElement.id === "user-input") {
          return;
        }
        if (lastKeyPress.key !== key && 
            parseInt(key) >= this.appSettings.keyboardRange.minKey && 
            parseInt(key) <= this.appSettings.keyboardRange.maxKey) {
          if (lastKeyPress.key !== null) {
            const text = document.getElementById("user-input").value;
            // Fix: Get the scale pitch value, not the index
            const scalePitchValue = this.state.scales[this.state.currentScale][parseInt(lastKeyPress.key) - 1];
            this.fetchFromBackendReverse(text, scalePitchValue);
            const keyElement = document.getElementById(this.convertSharpToFlat(lastKeyPress.note.toNote()));
            if (keyElement) {
              keyElement.classList.remove("highlight");
            }
          }
          lastKeyPress.key = key;
          lastKeyPress.time = performance.now();
          const pitch = this.convertToScale([(parseInt(key) - 1) * this.audioSettings.surprisalDivisor]);
          const note = Tone.Frequency(this.audioSettings.baseNote).transpose(pitch);
          lastKeyPress.note = note;
          const adjustedDuration = this.getSamplerAdjustedDuration(this.audioSettings.noteDuration);
          this.state.toneOutput.triggerAttackRelease(note, adjustedDuration, Tone.now());
          const keyElement = document.getElementById(this.convertSharpToFlat(lastKeyPress.note.toNote()));
          if (keyElement) {
            keyElement.classList.add("highlight");
          }
        } else if (action === 'up' && lastKeyPress.key === key) {
          const text = document.getElementById("user-input").value;
          // Fix: Get the scale pitch value, not the index
          const scalePitchValue = this.state.scales[this.state.currentScale][parseInt(key) - 1];
          this.fetchFromBackendReverse(text, scalePitchValue);
          const keyElement = document.getElementById(this.convertSharpToFlat(lastKeyPress.note.toNote()));
          if (keyElement) {
            keyElement.classList.remove("highlight");
          }
          lastKeyPress.key = null;
          lastKeyPress.time = null;
        }
      } catch (error) {
        ErrorHandler.logError(error, 'keyboard event handling');
      }
    }, this.uiSettings.debounceDelay);
    
    window.onkeydown = (event) => {
      try {
        if (document.activeElement.id === "user-input") {
          return;
        }
        if (lastKeyPress.key !== event.key && 
            parseInt(event.key) >= this.appSettings.keyboardRange.minKey && 
            parseInt(event.key) <= this.appSettings.keyboardRange.maxKey) {
          debouncedKeyHandler(event.key, 'down');
        }
      } catch (error) {
        ErrorHandler.logError(error, 'keydown');
      }
    };
    
    window.onkeyup = (event) => {
      try {
        if (document.activeElement.id === "user-input" || lastKeyPress.key === null) {
          return;
        }
        if (lastKeyPress.key === event.key) {
          debouncedKeyHandler(event.key, 'up');
        }
      } catch (error) {
        ErrorHandler.logError(error, 'keyup');
      }
    };
  }

  setupExamples() {
    document.getElementById("ex1")?.addEventListener("click", function() {
      try {
        document.getElementById("ex1_hidden")?.classList.remove("hiding");
        document.getElementById("ex1")?.classList.add("hiding");
      } catch (error) {
        ErrorHandler.logError(error, 'ex1 click');
      }
    });
    
    document.getElementById("ex2")?.addEventListener("click", function() {
      try {
        document.getElementById("ex2_hidden")?.classList.remove("hiding");
        document.getElementById("ex2")?.classList.add("hiding");
      } catch (error) {
        ErrorHandler.logError(error, 'ex2 click');
      }
    });
    
    document.getElementById("ex1_hidden")?.addEventListener("click", function() {
      try {
        document.getElementById("ex1_hidden")?.classList.add("hiding");
        document.getElementById("ex1")?.classList.remove("hiding");
      } catch (error) {
        ErrorHandler.logError(error, 'ex1_hidden click');
      }
    });
    
    document.getElementById("ex2_hidden")?.addEventListener("click", function() {
      try {
        document.getElementById("ex2_hidden")?.classList.add("hiding");
        document.getElementById("ex2")?.classList.remove("hiding");
      } catch (error) {
        ErrorHandler.logError(error, 'ex2_hidden click');
      }
    });

    document.getElementById("ex1_demo")?.addEventListener("click", () => { 
      try {
        const userInput = document.getElementById("user-input");
        if (userInput) {
          userInput.value = "the man fed the cat some tuna.";
          const text = userInput.value;
          this.fetchFromBackend(text);
          window.scrollTo({
            top: 0,
            behavior: "smooth"
          });
        }
      } catch (error) {
        ErrorHandler.logError(error, 'ex1_demo click');
        ErrorHandler.showError(`Failed to load example: ${error.message}`);
      }
    });

    document.getElementById("ex2_demo")?.addEventListener("click", () => { 
      try {
        const userInput = document.getElementById("user-input");
        if (userInput) {
          userInput.value = "the lawyer presented the cat with a lawsuit.";
          const text = userInput.value;
          this.fetchFromBackend(text);
          window.scrollTo({
            top: 0,
            behavior: "smooth"
          });
        }
      } catch (error) {
        ErrorHandler.logError(error, 'ex2_demo click');
        ErrorHandler.showError(`Failed to load example: ${error.message}`);
      }
    });
  }

  // Main initialization method
  async init() {
    // Update text limits from backend first
    await ErrorHandler.updateTextLimit();
    
    // Initialize scales first, then audio, then UI
    this.state.initScales();
    this.initAudio();
    this.setupInstruments();
    this.setupScales();
    this.setupModels();
    this.setupKeyboard();
    this.setupExamples();
    this.setupSubmitButton();
    this.setupKeyboardEvents();
    AccessibilityUtils.addKeyboardNavigation();
    AccessibilityUtils.updateAriaLabels();
    this.state.updateIndicator();
    
    // Setup global error handling
    ErrorHandler.setupGlobalErrorHandling();
  }
} 