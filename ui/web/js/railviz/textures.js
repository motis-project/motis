var RailViz = RailViz || {};

RailViz.Textures = (function() {

  function createCircle(fillColor, borderColor, baseBorderThickness) {
    return function(size) {
      const borderThickness = baseBorderThickness / 64 * size;
      const rad = size / 2 - borderThickness;
      var cv = document.createElement('canvas');
      cv.width = size;
      cv.height = size;
      var ctx = cv.getContext('2d', {alpha: true});
      ctx.beginPath();
      ctx.arc(
          rad + borderThickness, rad + borderThickness, rad, 0, 2 * Math.PI,
          false);
      ctx.fillStyle = 'rgba(' + fillColor[0] + ',' + fillColor[1] + ',' +
          fillColor[2] + ',' + fillColor[3] + ')';
      ctx.fill();
      ctx.lineWidth = borderThickness;
      ctx.strokeStyle = 'rgba(' + borderColor[0] + ',' + borderColor[1] + ',' +
          borderColor[2] + ',' + borderColor[3] + ')';
      ctx.stroke();
      return cv;
    };
  }

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

  return {createCircle: createCircle, createTrain: createTrain};
})();
