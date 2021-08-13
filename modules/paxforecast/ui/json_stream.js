class JSFrame {
  constructor(skip, arr = null, obj = null) {
    this.skip = skip;
    this.arr = skip ? null : arr;
    this.obj = skip ? null : obj;
    this.currentKey = null;
  }
}

class JSONStream {
  constructor(skipInitial = true) {
    // options
    this.progressUpdateStep = 1024 * 1024;

    // callbacks
    this.onprogress = null;
    this.onkey = null;
    this.onobject = null;
    this.onarray = null;

    // internals
    this.parser = clarinet.parser();

    this.currentFrame = new JSFrame(skipInitial);
    this.frames = [];
    this.captureNext = false;

    [
      "onopenobject",
      "oncloseobject",
      "onopenarray",
      "onclosearray",
      "onkey",
      "onvalue",
    ].forEach((e) => {
      this.parser[e] = this["_" + e].bind(this);
    });
  }

  async parseBlob(blob) {
    const stream = blob.stream();
    const reader = stream.getReader();
    const decoder = new TextDecoder("utf-8");
    const fileSize = blob.size;
    let progress = 0;
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          this.parser.close();
          break;
        }
        this.parser.write(decoder.decode(value));
        progress += value.length;
        if (this.onprogress) {
          this.onprogress(progress, fileSize);
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  currentDepth() {
    return this.frames.length;
  }

  // internals

  pushFrame(arr, obj) {
    const skip = this.currentFrame.skip && !this.captureNext;
    this.frames.push(this.currentFrame);
    this.currentFrame = new JSFrame(skip, arr, obj);
    this.captureNext = false;
    return this.currentFrame;
  }

  popFrame() {
    this.currentFrame = this.frames.pop();
    return this.currentFrame;
  }

  finish(parent, child, type) {
    if (child.skip) {
      return;
    }
    if (parent.arr) {
      parent.arr.push(child[type]);
    } else if (parent.obj && parent.currentKey) {
      parent.obj[parent.currentKey] = child[type];
    } else {
      const handler = this["on" + (type === "arr" ? "array" : "object")];
      if (handler) {
        handler(parent.currentKey, child[type]);
      }
    }
  }

  _onopenobject(key) {
    this.pushFrame(null, {});
    this._onkey(key);
  }

  _oncloseobject() {
    const child = this.currentFrame;
    const parent = this.popFrame();
    this.finish(parent, child, "obj");
  }

  _onopenarray() {
    this.pushFrame([], {});
  }

  _onclosearray() {
    const child = this.currentFrame;
    const parent = this.popFrame();
    this.finish(parent, child, "arr");
  }

  _onkey(key) {
    if (this.currentFrame.skip) {
      if (this.onkey && this.onkey(key)) {
        this.captureNext = true;
      } else {
        return;
      }
    }
    this.currentFrame.currentKey = key;
  }

  _onvalue(val) {
    if (this.currentFrame.skip) {
      if (this.captureNext) {
        this.captureNext = false;
        if (this.onvalue) {
          this.onvalue(this.currentFrame.currentKey, val);
        }
      }
    } else if (this.currentFrame.arr) {
      this.currentFrame.arr.push(val);
    } else if (this.currentFrame.obj && this.currentFrame.currentKey) {
      this.currentFrame.obj[this.currentFrame.currentKey] = val;
    }
  }
}
