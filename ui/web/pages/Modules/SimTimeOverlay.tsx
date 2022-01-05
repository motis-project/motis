import React from "react";

export const SimTimeOverlay: React.FC = (props) => {
    return (
        <div class="sim-time-overlay">
            <div id="railviz-loading-spinner" class="">
                <div class="spinner">
                    <div class="bounce1"></div>
                    <div class="bounce2"></div>
                    <div class="bounce3"></div>
                </div>
            </div>
            <div class="permalink" title="Permalink">
                <a href="#/railviz/50.65753/9.479082/6/0/0/1640866963">
                <i class="icon">link</i>
                </a>
            </div>
            <div class="sim-icon" title="Simulationsmodus aktiv">
                <i class="icon">warning</i>
            </div>
            <div class="time" id="sim-time-overlay">30.12.2021 13:22:43</div>
        </div>
    );
};