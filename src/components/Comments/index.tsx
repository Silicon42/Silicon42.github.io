import React from "react";
import Giscus from "@giscus/react";
import { useColorMode } from "@docusaurus/theme-common";

export default function Comments(): JSX.Element {
  const { colorMode } = useColorMode();

  return (
    <div>
      <Giscus
        id="comments"
		repo="Silicon42/Silicon42.github.io"
		repoId="R_kgDONrcYnQ"
		category="Announcements"
		categoryId="DIC_kwDONrcYnc4CmE7r"
		mapping="pathname"
		strict="1"
		reactions-enabled="1"
		emit-metadata="0"
		input-position="top"
		theme="preferred_color_scheme"
		lang="en"
		loading="lazy"
      />
    </div>
  );
}