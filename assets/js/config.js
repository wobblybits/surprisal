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
  // choir: {
  //   portamento: 0.01,
  //   oscillator: {
  //     type: "fatsine",
  //     modulationType: "sine4",
  //     partials: [1, 1, 0.9, 1, 1.2, 0.4, 0.8, 0.7],
  //     partialCount: 2,
  //     spread: 10,
  //     phase: Math.PI * 2,
  //     count: 4,
  //   },
  //   envelope: {
  //     attack: 2,
  //     decay: 0,
  //     sustain: 1,
  //     release: 2,
  //   },
  // },
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
 
export const models = ["gpt2", "smollm", "nano mistral", "smol llama", "qwen", "flan"];
