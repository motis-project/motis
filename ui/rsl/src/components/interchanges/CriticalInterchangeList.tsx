import { useInfiniteQuery } from "@tanstack/react-query";
import { useAtom } from "jotai";
import { ReactElement, useCallback } from "react";

import { PaxMonCriticalInterchangesRequest } from "@/api/protocol/motis/paxmon";

import { queryKeys, sendPaxMonCriticalInterchangesRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

function CriticalInterchangeList(): ReactElement {
  const [universe] = useAtom(universeAtom);

  const baseRequest: PaxMonCriticalInterchangesRequest = {
    universe,
    filter_interval: { begin: 0, end: 0 },
    sort_by: "TotalDelayIncrease",
    max_results: 100,
    skip_first: 0,
  };

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetching,
    refetch,
    /*
    error,
    isFetching,
    isFetchingNextPage,
    status,
    isLoading,
    isStale,
    isPreviousData,
    */
  } = useInfiniteQuery(
    queryKeys.criticalInterchanges(baseRequest),
    ({ pageParam = 0 }) =>
      sendPaxMonCriticalInterchangesRequest({
        ...baseRequest,
        skip_first: pageParam as number,
      }),
    {
      getNextPageParam: (lastPage) =>
        lastPage.remaining_interchanges > 0 ? lastPage.next_skip : undefined,
      refetchOnWindowFocus: true,
      keepPreviousData: true,
      staleTime: 60000,
    },
  );

  const loadMore = useCallback(async () => {
    if (hasNextPage) {
      return await fetchNextPage();
    }
  }, [fetchNextPage, hasNextPage]);

  return <div>TODO</div>;
}

export default CriticalInterchangeList;
