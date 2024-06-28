<script type="ts">
	let c = {
		type: 'FeatureCollection',
		features: [
			{
				type: 'Feature',
				properties: {
					mode: 'transport',
					direction: 'Dieburg Bahnhof',
                    provider: {
                        agency_id: 
                    },
					stops: [
						{
							name: 'Breuberg-Wald-Amorbach An der alten Brücke',
							departure: '2024-06-25T15:08:00-02:00'
						},
						{
							name: 'Groß-Umstadt-Dorndiel Mitte',
							arrival: '2024-06-25T15:12:00-02:00',
							departure: '2024-06-25T15:12:00-02:00'
						},
						{
							name: 'Groß-Umstadt Mitte Bahnhof',
							arrival: '2024-06-25T15:12:00-02:00'
						}
					]
				},
				geometry: {
					type: 'PolyLine',
					coordinates: []
				}
			},
			{
				type: 'Feature',
				properties: {
					mode: 'transport',
					direction: 'Dieburg Bahnhof',
					stops: [
						{
							name: 'Breuberg-Wald-Amorbach An der alten Brücke',
							departure: '2024-06-25T15:08:00-02:00'
						},
						{
							name: 'Groß-Umstadt-Dorndiel Mitte',
							arrival: '2024-06-25T15:12:00-02:00',
							departure: '2024-06-25T15:12:00-02:00'
						},
						{
							name: 'Groß-Umstadt Mitte Bahnhof',
							arrival: '2024-06-25T15:12:00-02:00'
						}
					]
				},
				geometry: {
					type: 'PolyLine',
					coordinates: []
				}
			}
		]
	};
</script>

<div class="connection-journey" id="connection-journey">
	<div class="train-detail train-class-10">
		<div class="top-border"></div>
		<div>
			<div class="train-box train-class-10 with-tooltip" data-tooltip="Betreiber: Fa. Jungermann">
				<svg class="train-icon">
					<use xlink:href="#bus"></use>
				</svg>
				<span class="train-name">Bus GU2</span>
			</div>
		</div>
		<div class="first-stop">
			<div class="stop past">
				<div class="timeline train-color-border"></div>
				<div class="time"><span class="past">15:08</span></div>
				<div class="delay"></div>
				<div class="station">
					<span>Breuberg-Wald-Amorbach An der alten Brücke</span>
				</div>
			</div>
		</div>
		<div class="direction past">
			<div class="timeline train-color-border"></div>
			<i class="icon">arrow_forward</i>Dieburg Bahnhof
		</div>
		<div class="intermediate-stops-toggle clickable past">
			<div class="timeline-container">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
			</div>
			<div class="expand-icon"><i class="icon">expand_less</i><i class="icon">expand_more</i></div>
			<span>Fahrt 8 Stationen (22min)</span>
		</div>
		<div class="intermediate-stops collapsed">
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:12</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt-Dorndiel Mitte</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:13</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt Dorndiel Friedhof</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:16</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt-Raibach Am alten Steinbruch</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:19</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt-Raibach Pumpstation</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:21</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt-Raibach Alte Schule</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:24</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt Gruberhof</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:25</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt Stadtfriedhof</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:27</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt Pfälzer Schloß</span></div>
			</div>
		</div>
		<div class="last-stop">
			<div class="stop past">
				<div class="timeline train-color-border"></div>
				<div class="time"><span class="past">15:30</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt Mitte Bahnhof</span></div>
			</div>
		</div>
	</div>
	<div class="train-detail train-class-6">
		<div class="top-border"></div>
		<div>
			<div class="train-box train-class-6 with-tooltip" data-tooltip="Betreiber: VIAS Rail GmbH">
				<svg class="train-icon">
					<use xlink:href="#train"></use>
				</svg><span class="train-name">RB86</span>
			</div>
		</div>
		<div class="train-top-line"><span>2min Fußweg</span></div>
		<div class="train-dep-track">Gleis 1</div>
		<div class="first-stop">
			<div class="stop past">
				<div class="timeline train-color-border"></div>
				<div class="time"><span class="past">15:44</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt Mitte Bahnhof</span></div>
				<div class="track"><span>Gleis 1</span></div>
			</div>
		</div>
		<div class="direction past">
			<div class="timeline train-color-border"></div>
			<i class="icon">arrow_forward</i>Hanau Hauptbahnhof
		</div>
		<div class="intermediate-stops-toggle clickable past">
			<div class="timeline-container">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
			</div>
			<div class="expand-icon"><i class="icon">expand_less</i><i class="icon">expand_more</i></div>
			<span>Fahrt 6 Stationen (38min)</span>
		</div>
		<div class="intermediate-stops collapsed">
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:48</span></div>
				<div class="delay"></div>
				<div class="station"><span>Groß-Umstadt-Klein-Umstadt Bahnhof</span></div>
				<div class="track"><span>Gleis 1</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">15:51</span></div>
				<div class="delay"></div>
				<div class="station"><span>Babenhausen Langstadt Bahnhof</span></div>
				<div class="track"><span>Gleis 1</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">16:01</span></div>
				<div class="delay"></div>
				<div class="station"><span>Babenhausen Bahnhof</span></div>
				<div class="track"><span>Gleis 3</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">16:08</span></div>
				<div class="delay"></div>
				<div class="station"><span>Seligenstadt Bahnhof</span></div>
				<div class="track"><span>Gleis 1</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">16:14</span></div>
				<div class="delay"></div>
				<div class="station"><span>Hainburg Hainstadt Bahnhof</span></div>
				<div class="track"><span>Gleis 2</span></div>
			</div>
			<div class="stop past">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 100%;"></div>
				<div class="time"><span class="past">16:17</span></div>
				<div class="delay"></div>
				<div class="station"><span>Hanau Klein-Auheim Bahnhof</span></div>
				<div class="track"><span>Gleis 1</span></div>
			</div>
		</div>
		<div class="last-stop">
			<div class="stop past">
				<div class="timeline train-color-border"></div>
				<div class="time"><span class="past">16:22</span></div>
				<div class="delay"></div>
				<div class="station"><span>Hanau Hauptbahnhof</span></div>
				<div class="track"><span>Gleis 104</span></div>
			</div>
			<div class="train-arr-track">Gleis 104</div>
		</div>
	</div>
	<div class="train-detail train-class-1">
		<div class="top-border"></div>
		<div>
			<div
				class="train-box train-class-1 with-tooltip"
				data-tooltip="Betreiber: SBB
Linie: 12"
			>
				<svg class="train-icon">
					<use xlink:href="#train"></use>
				</svg><span class="train-name">ICE 00278</span>
			</div>
		</div>
		<div class="train-top-line"><span>Ankunft Gleis 104, 2min Fußweg</span></div>
		<div class="train-dep-track">Gleis 7</div>
		<div class="first-stop">
			<div class="stop past">
				<div class="timeline train-color-border"></div>
				<div class="time"><span class="past">16:30</span></div>
				<div class="delay"></div>
				<div class="station"><span>Hanau Hauptbahnhof</span></div>
				<div class="track"><span>Gleis 7</span></div>
			</div>
		</div>
		<div class="direction past">
			<div class="timeline train-color-border"></div>
			<i class="icon">arrow_forward</i>S Ostbahnhof (Berlin)
		</div>
		<div class="intermediate-stops-toggle clickable future">
			<div class="timeline-container">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 13%;"></div>
			</div>
			<div class="expand-icon"><i class="icon">expand_less</i><i class="icon">expand_more</i></div>
			<span>Fahrt 7 Stationen (3h 59min)</span>
		</div>
		<div class="intermediate-stops collapsed">
			<div class="stop future">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 0%;"></div>
				<div class="time"><span class="future">17:12</span></div>
				<div class="delay"></div>
				<div class="station"><span>Fulda Bahnhof</span></div>
				<div class="track"><span>Gleis 6</span></div>
			</div>
			<div class="stop future">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 0%;"></div>
				<div class="time"><span class="future">17:45</span></div>
				<div class="delay"></div>
				<div class="station"><span>Kassel Bahnhof Wilhelmshöhe</span></div>
			</div>
			<div class="stop future">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 0%;"></div>
				<div class="time"><span class="future">18:06</span></div>
				<div class="delay"></div>
				<div class="station"><span>Göttingen</span></div>
			</div>
			<div class="stop future">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 0%;"></div>
				<div class="time"><span class="future">18:36</span></div>
				<div class="delay"></div>
				<div class="station"><span>Hildesheim Hauptbahnhof</span></div>
			</div>
			<div class="stop future">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 0%;"></div>
				<div class="time"><span class="future">19:01</span></div>
				<div class="delay"></div>
				<div class="station"><span>Braunschweig, Hauptbahnhof/Viewegs Garten</span></div>
			</div>
			<div class="stop future">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 0%;"></div>
				<div class="time"><span class="future">19:19</span></div>
				<div class="delay"></div>
				<div class="station"><span>Wolfsburg, Hauptbahnhof</span></div>
			</div>
			<div class="stop future">
				<div class="timeline train-color-border bg"></div>
				<div class="timeline train-color-border progress" style="height: 0%;"></div>
				<div class="time"><span class="future">20:17</span></div>
				<div class="delay"></div>
				<div class="station"><span>S Spandau Bhf (Berlin)</span></div>
				<div class="track"><span>Gleis 6</span></div>
			</div>
		</div>
		<div class="last-stop">
			<div class="stop future">
				<div class="timeline train-color-border"></div>
				<div class="time"><span class="future">20:29</span></div>
				<div class="delay"></div>
				<div class="station"><span>S+U Berlin Hauptbahnhof</span></div>
				<div class="track"><span>Gleis 11</span></div>
			</div>
			<div class="train-arr-track">Gleis 11</div>
		</div>
	</div>
</div>
