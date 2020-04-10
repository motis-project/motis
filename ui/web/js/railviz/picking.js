var RailViz = RailViz || {};

RailViz.Picking = (function() {

    function pickIdToColor(id) {
        return [
            id & 255,
            (id >>> 8) & 255,
            (id >>> 16) & 255
        ];
    }

    function colorToPickId(color) {
        if (!color || color[3] === 0) {
            return null;
        } else {
            return color[0] + (color[1] << 8) + (color[2] << 16);
        }
    }

    return {
        pickIdToColor: pickIdToColor,
        colorToPickId: colorToPickId
    };

})();
