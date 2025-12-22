<script lang="ts">
	import type { Error } from '@motis-project/motis-client';
	import { CircleAlert, SearchX, ServerCrash } from '@lucide/svelte';

	let {
		e
	}: {
		e: Error | string;
	} = $props();

	let error = (typeof e == 'string' ? { status: 404, message: e } : e) as Error;
	const getErrorType = (status: number) => {
		switch (status) {
			case 400:
				return 'Bad Request';
			case 404:
				return 'Not Found';
			case 500:
				return 'Internal Server Error';
			default:
				return 'Unkown';
		}
	};

	const getErrorIcon = (status: number) => {
		switch (status) {
			case 400:
				return CircleAlert;
			case 404:
				return SearchX;
			case 500:
				return ServerCrash;
			default:
				return SearchX;
		}
	};
	let Icon = $state(getErrorIcon(error.status));
</script>

<div
	class="p-4 mx-auto my-4 w-80 flex flex-col items-center gap-5 rounded-lg border border-destructive/20"
>
	<div class="flex items-center gap-4 mx-auto">
		<Icon class="h-7 w-7 text-destructive" />
		<h2 class="text-xl font-semibold text-destructive">
			{error.status}
			{getErrorType(error.status)}
		</h2>
	</div>
	<p class="text-lg text-muted-foreground mb-3">
		{error.message}
	</p>
</div>
