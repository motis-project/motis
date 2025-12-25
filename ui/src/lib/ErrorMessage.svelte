<script lang="ts">
	import { CircleAlert, SearchX, ServerCrash } from '@lucide/svelte';

	let {
		message,
		status
	}: {
		message: string;
		status: number | undefined;
	} = $props();

	const getErrorType = (status: number) => {
		switch (status) {
			case 400:
				return 'Bad Request';
			case 404:
				return 'Not Found';
			case 500:
				return 'Internal Server Error';
			case 422:
				return 'Unprocessable Entity';
			default:
				return 'Unknown';
		}
	};

	const getErrorIcon = (status: number) => {
		switch (status) {
			case 400:
				return CircleAlert;
			case 404:
				return SearchX;
			case 422:
			case 500:
				return ServerCrash;
			default:
				return SearchX;
		}
	};
	let Icon = $state(getErrorIcon(status ?? 404));
</script>

<div
	class="p-4 mx-auto my-4 w-96 flex flex-col items-center gap-5 rounded-lg border border-destructive/20"
>
	<div class="flex items-center gap-4 mx-auto">
		<Icon class="h-7 w-7 text-destructive" />
		<h2 class="text-xl font-semibold text-destructive">
			{status}
			{getErrorType(status ?? 404)}
		</h2>
	</div>
	<p class="text-lg text-muted-foreground mb-3">
		{message}
	</p>
</div>
