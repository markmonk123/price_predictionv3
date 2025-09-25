/**
 * Lightweight in-project QuickFIX stub used when the native quick-fix package
 * is unavailable. It supplies the minimal surface that the platform relies on
 * for simulation and logging. Replace with the real binding when integrating
 * against a production FIX engine.
 */

class FileStoreFactory {
  constructor(directory) {
    this.directory = directory;
  }
}

class FileLogFactory {
  constructor(directory) {
    this.directory = directory;
  }
}

class Dictionary {
  constructor(configuration = {}) {
    this.configuration = configuration;
  }
}

class Initiator {
  constructor(storeFactory, logFactory, dictionary, application = {}) {
    this.storeFactory = storeFactory;
    this.logFactory = logFactory;
    this.dictionary = dictionary;
    this.application = application;
    this.started = false;
  }

  start() {
    this.started = true;
    if (typeof this.application.onCreate === 'function') {
      this.application.onCreate('SIMULATED_SESSION');
    }
    if (typeof this.application.onLogon === 'function') {
      this.application.onLogon('SIMULATED_SESSION');
    }
  }
}

module.exports = {
  FileStoreFactory,
  FileLogFactory,
  Dictionary,
  Initiator
};
