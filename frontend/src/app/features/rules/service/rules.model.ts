export interface RulesConfig {
  // Search providers
  searxng: boolean;
  duckduckgo: boolean;
  tavily: boolean;
  serper: boolean;

  // Extraction methods
  tavilyExtract: boolean;
  localExtract: boolean;

  // Advanced modes
  advanceSearch: boolean;
  advanceExtract: boolean;

  // Custom limits (optional overrides)
  webSearchLimit?: number;
  memorySearchLimit?: number;
  historyLimit?: number;
  fileUploadMaxChars?: number;

  // Custom AI behavior
  customInstructions?: string;

  // Reasoning generation (default: false)
  reasoningEnabled?: boolean;
}

export const DEFAULT_RULES: RulesConfig = {
  // Search providers (Free providers enabled by default)
  searxng: true,
  duckduckgo: true,
  tavily: false, // Requires API key
  serper: false, // Requires API key

  // Extraction methods
  tavilyExtract: false, // Requires API key
  localExtract: true, // Free, enabled by default

  // Advanced modes (disabled by default)
  advanceSearch: false,
  advanceExtract: false,

  // Custom instructions (empty by default)
  customInstructions: '',

  // Reasoning generation (disabled by default)
  reasoningEnabled: false,
};

// Helper function to validate rules
export function validateRulesConfig(rules: Partial<RulesConfig>): RulesConfig {
  return {
    ...DEFAULT_RULES,
    ...rules,
  };
}

// Check if any search provider is enabled
export function hasSearchProviderEnabled(rules: RulesConfig): boolean {
  return rules.searxng || rules.duckduckgo || rules.tavily || rules.serper;
}

// Check if any extraction method is enabled
export function hasExtractionEnabled(rules: RulesConfig): boolean {
  return rules.tavilyExtract || rules.localExtract;
}

// Get list of enabled search providers
export function getEnabledSearchProviders(rules: RulesConfig): string[] {
  const providers: string[] = [];
  if (rules.searxng) providers.push('SearXNG');
  if (rules.duckduckgo) providers.push('DuckDuckGo');
  if (rules.tavily) providers.push('Tavily');
  if (rules.serper) providers.push('Serper');
  return providers;
}

// Get list of enabled extraction methods
export function getEnabledExtractionMethods(rules: RulesConfig): string[] {
  const methods: string[] = [];
  if (rules.tavilyExtract) methods.push('Tavily Extract');
  if (rules.localExtract) methods.push('Local Extract');
  return methods;
}
