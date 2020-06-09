var RailViz = RailViz || {};

RailViz.Textures = (function() {

  function createTrain() {
    return function(size) {
      const border = (2 / 64) * size;
      const padding = (size - (size / 2)) / 2 + border;
      const innerSize = size - (2 * padding);
      const mid = size / 2;
      const rad = innerSize / 3.5;
      var cv = document.createElement('canvas');
      cv.width = size;
      cv.height = size;
      var ctx = cv.getContext('2d', {alpha: true});

      ctx.beginPath();

      ctx.arc(padding + rad, mid, rad, 1 / 2 * Math.PI, 3 / 2 * Math.PI, false);

      ctx.bezierCurveTo(
          padding + rad + rad, mid - rad, size - padding, mid, size - padding,
          mid);
      ctx.bezierCurveTo(
          size - padding, mid, padding + rad + rad, mid + rad, padding + rad,
          mid + rad);

      ctx.closePath();

      ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
      ctx.fill();
      ctx.lineWidth = border;
      ctx.strokeStyle = 'rgba(160, 160, 160, 1.0)';
      ctx.stroke();
      return cv;
    };
  }

  function createShield(opt) {
    const d = 32

    const cv = document.createElement('canvas');
    cv.width = d;
    cv.height = d;
    const ctx = cv.getContext('2d');

    // coord of the line (front = near zero, back = opposite)
    const l_front = 1;
    const l_back = d-1;

    // coord start of the arc
    const lr_front = l_front + 2;
    const lr_back = l_back - 2;

    // control point of the arc
    const lp_front = l_front + 1;
    const lp_back = l_back - 1;

    let p = new Path2D();
    p.moveTo(lr_front, l_front);

    // top line
    p.lineTo(lr_back, l_front);
    // top right corner
    p.bezierCurveTo(lp_back, lp_front, lp_back, lp_front, l_back, lr_front);
    // right line
    p.lineTo(l_back, lr_back);
    // bottom right corner
    p.bezierCurveTo(lp_back, lp_back, lp_back, lp_back, lr_back, l_back);
    // bottom line
    p.lineTo(lr_front, l_back);
    // bottom left corner
    p.bezierCurveTo(lp_front, lp_back, lp_front, lp_back, l_front, lr_back);
    // left line
    p.lineTo(l_front, lr_front);
    // top left corner
    p.bezierCurveTo(lp_front, lp_front, lp_front, lp_front, lr_front, l_front);

    p.closePath();

    ctx.fillStyle = opt.fill;
    ctx.fill(p);
    ctx.strokeStyle = opt.stroke;
    ctx.stroke(p);

    return [
      ctx.getImageData(0, 0, d, d),
      {
        content: [lr_front, lr_front, lr_back, lr_back],
        stretchX: [[lr_front, lr_back]],
        stretchY: [[lr_front, lr_back]]
      }
    ];
  }

  function createHexShield(opt) {
    const d = 64

    const cv = document.createElement('canvas');
    cv.width = d;
    cv.height = d;
    const ctx = cv.getContext('2d');

    // coord of the line (front = near zero, back = opposite)
    const l_front = 3;
    const l_back = d-3;

    // corner points
    const lp_front = l_front + 8;
    const lp_half = d / 2;
    const lp_back = l_back - 8;

    let p = new Path2D();
    p.moveTo(lp_half, l_front); // top
    p.lineTo(l_back, lp_front); // top right
    p.lineTo(l_back, lp_back); // bot right
    p.lineTo(lp_half, l_back); // bot
    p.lineTo(l_front, lp_back); // bot left
    p.lineTo(l_front, lp_front); // top left
    p.closePath();

    ctx.fillStyle = opt.fill;
    ctx.fill(p);
    ctx.lineWidth = 3;
    ctx.strokeStyle = opt.stroke;
    ctx.stroke(p);

    return [
      ctx.getImageData(0, 0, d, d),
      {content: [l_front + 4, lp_front + 2, l_back - 4, lp_back - 2]}
    ];
  }

  return {
    createTrain: createTrain,
    createShield: createShield,
    createHexShield: createHexShield
  };
})();
