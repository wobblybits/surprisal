export const presets = {
  xylophone: {
    portamento: 0.0,
    oscillator: {
      type: "sine",
      partials: [1, 0, 2, 0, 3],
    },
    envelope: {
      attack: 0.001,
      decay: 1.2,
      sustain: 0,
      release: 1.2,
    },
  },
  flute: {
    portamento: 0.0,
    oscillator: {
      type: "fatsine",
      partials: [1, 1, 0.1, 0.2, 0.15, 0.01, 0.01, 0.01],
      partialCount: 8,
      modulationType: "sine4",
      phase: 1,
      spread: 20,
      count: 2,
    },
    envelope: {
      attack: 0.5,
      decay: 2,
      sustain: 1,
      release: 1,
    },
  },
  violin: {
    portamento: 0.01,
    oscillator: {
      type: "fmsquare",
      modulationType: "sawtooth",
      harmonicity: 4,
      partials: [1, 0.6, 0.6, 0.7, 0.5, 0.2, 0.5, 0.15],
      partialCount: 8,
    },
    envelope: {
      attack: 2.2,
      decay: 0.3,
      sustain: 1,
      release: 1.2,
    },
  },
  theremin: {
    portamento: 0.07,
    oscillator: {
      type: "sine",
      partials: [1, 0.1],
    },
    envelope: {
      attack: 0.1,
      decay: 1.2,
      sustain: 1,
      release: 1,
    },
  },
  piano: {
    portamento: 0.0,
    oscillator: {
      type: "amsine",
      modulationType: "triangle",
      harmonicity: 0.5,
      partials: [1, 0.5, 0.3, 0.2, 0.1],
    },
    envelope: {
      attack: 0.001,
      decay: 2,
      sustain: 0,
      release: 3,
    },
  },
  cat: {
    sampler: true,
  }
};

export const scaleIntervals = {
  heptatonic: {
    "major-7": [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23],
    "minor-7": [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22],
  },
  pentatonic: {
    "major-5": [0, 2, 4, 7, 9, 12, 14, 16, 19, 21],
    "minor-5": [0, 3, 5, 8, 10, 12, 15, 17, 20, 22],
  },
  atonal: {
    chromatic: [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23,
    ],
    continuous: [],
  },
};

export const notes = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"];
 
export const models = ["gpt2", "smollm", "nano mistral", "qwen", "flan"];

// Application Settings - Core app behavior and limits
export const applicationSettings = {
  maxTextLength: 1000,  // Default - will sync with backend via API
  modelNameDisplayLength: 10,
  settingsIndices: {
    instrument: 0,
    scale: 1,
    model: 2
  },
  keyboardRange: {
    minKey: 1,
    maxKey: 9
  },
  defaultSettings: ["THEREMIN", "MAJOR-7", "GPT2"],
  defaultVector: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
};

// Audio Settings - Musical and audio processing parameters
export const audioSettings = {
  volumeScaling: 8,
  surprisalDivisor: 2,
  baseNote: "C3",
  defaultVolume: 0.5,
  transportStart: 0,
  noteDuration: "8n",
  samplerMinDuration: 1, // Minimum duration for sampler instruments in seconds
  continuousPitch: {
    baseNote: 48,
    referenceNote: 69,
    baseFrequency: 440,
    c3Frequency: 130,
    pitchCalculation: {
      semitonesPerOctave: 12
    }
  },
  keyboard: {
    numOctaves: 2,
    startingOctave: 3,
    startingNote: "C",
    semitonesPerOctave: 12,
    spacing: {
      whiteKeys: 7,
      numKeysPastOctave: 0
    }
  },
  playback: {
    highlightDurationMultiplier: 900,
    timeoutConversionMs: 1000
  }
};

// UI Settings - User interface behavior and styling
export const uiSettings = {
  debounceDelay: 50,
  errorDisplayDuration: 5000,
  announcementDelay: 1000,
  selectionReset: {
    startPos: 0,
    endPos: 0
  },
  zIndex: {
    overlay: 1000,
    error: 1000
  },
  positioning: {
    center: "50%",
    centerTransform: "translate(-50%, -50%)"
  },
  opacity: {
    loadingBackground: "rgba(0,0,0,0.8)"
  },
  spinner: {
    borderRadius: "50%",
    size: "40px",
    borderWidth: "4px",
    marginBottom: "10px"
  },
  boxShadow: {
    error: "0 2px 10px rgba(0,0,0,0.3)"
  }
};

// Validation Settings - Input validation parameters
export const validationSettings = {
  textInput: {
    maxLength: 1000, // Will be updated from backend
    minLength: 0
  },
  noteLength: {
    minLength: 1
  },
  arrayIndices: {
    noteChar: 1,
    noteOctave: 2
  }
};
