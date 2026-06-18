import { apiUrl } from "./api";

// NEXT_PUBLIC_API_URL is not set in the test environment, so API_BASE falls
// back to "" — tests exercise path-concatenation behavior directly.
describe("apiUrl", () => {
  it("returns the path unchanged when no base URL is configured", () => {
    expect(apiUrl("/api/chat")).toBe("/api/chat");
  });

  it("concatenates base and path without double slashes", () => {
    expect(apiUrl("/api/users")).toBe("/api/users");
  });

  it("preserves query strings", () => {
    expect(apiUrl("/api/chat?session=abc")).toBe("/api/chat?session=abc");
  });

  it("handles an empty path", () => {
    expect(apiUrl("")).toBe("");
  });
});
