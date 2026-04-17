<script lang="ts">
	import { CircleAlert, CircleCheck, SearchX, ServerCrash } from '@lucide/svelte';

	let {
		message,
		status
	}: {
		message: string;
		status: number | undefined;
	} = $props();

	const getErrorType = (status: number) => {
		switch (status) {
			case 200:
				return 'OK';
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
			case 200:
				return CircleCheck;
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
	class="p-4 my-4 mx-auto min-w-96 max-w-fit flex flex-col items-center gap-5 rounded-lg border border-destructive/20"
>
	<div class="flex items-center gap-4 mx-auto">
		<Icon class="h-7 w-7 text-destructive" />
		<h2 class="text-xl font-semibold text-destructive">
			{status}
			{getErrorType(status ?? 404)}
		</h2>
	</div>
	<p class="text-lg text-muted-foreground max-w-[40ch] break-words">
		{message}
	</p>
</div>
