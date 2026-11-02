# Moving Dermalis to React Native

The current UI is a single `frontend/index.html`. This is the plan for rebuilding it as a React Native app without losing the look.

## Recommended stack

| Need | Choice | Why |
|---|---|---|
| App framework | **Expo** (React Native) + **react-native-web** | One codebase for iOS, Android and web; no native tooling needed to start. |
| Styling | **NativeWind** (Tailwind classes for RN) | Same utility-class workflow on every platform; the tokens below drop straight into `tailwind.config.js`. |
| Hand-drawn shapes, doodles, textures | **react-native-svg** | The leaf logo, sprout, tape strips and paper grain are already SVG. |
| Motion | **react-native-reanimated** (or **Moti** on top of it) | Card tilt/lift, progress bar fill, pulsing "working" dot. |
| Fonts | `@expo-google-fonts/fraunces`, `/nunito`, `/caveat` | Same three fonts as the web page. |
| Image picker | `expo-image-picker` | Replaces the file input; also gives camera capture on phones. |

The backend API (`/analyze`, `/chat`) and its JSON shapes stay the same.

## Design tokens → `tailwind.config.js`

These mirror the CSS variables at the top of `frontend/index.html`.

```js
module.exports = {
  theme: {
    extend: {
      colors: {
        sage: { 100: "#E6EEE3", 300: "#B5C9B1", 500: "#7A9E7E", 700: "#4F6B53" },
        terracotta: { 300: "#E9B4A2", 500: "#C8664B", 700: "#9A4A35" },
        mustard: { 300: "#EFD58F", 500: "#D9A441" },
        navy: { 500: "#3E5170", 700: "#2B3A55" },
        paper: { DEFAULT: "#F6EFE3", light: "#FBF7F0" },
        clay: "#E8D9C4",
        ink: { DEFAULT: "#2A2620", muted: "#6E655A" },
      },
      fontFamily: {
        display: ["Fraunces_700Bold"],
        body: ["Nunito_400Regular"],
        hand: ["Caveat_500Medium"],
      },
    },
  },
};
```

React Native has no asymmetric `border-radius` shorthand or `::before`, so:
- **Hand-cut corners**: set the four `border*Radius` props individually (e.g. 18/24/16/22).
- **Offset "paper" shadow**: an absolutely positioned ink-colored `View` behind the card, offset 3-4px (more consistent than platform shadows).
- **Washi tape, paper grain**: small SVG components.

## Component mapping

| Web (`index.html`) | React Native |
|---|---|
| Three-column `main` grid | `View` with `flex-row` on wide screens (`useWindowDimensions`), stacked on phones |
| Progress sidebar (`renderProgress`) | `ProgressCard` component; the step list and time logic port as-is |
| Chat `#messages` | `FlatList` (inverted) of message bubbles |
| Product type tabs | Horizontal `ScrollView` of `Pressable` chips |
| Product cards | `Pressable` card component; rotation via `transform: [{ rotate: "-0.6deg" }]` |
| `<dialog>` product details | `Modal` (or a bottom sheet such as `@gorhom/bottom-sheet` on phones) |
| Undo toast | Small animated `View` pinned to the bottom |
| `localStorage` ratings | `@react-native-async-storage/async-storage` |
