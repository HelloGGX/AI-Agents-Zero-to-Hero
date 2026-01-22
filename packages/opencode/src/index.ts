import yargs from "yargs"
import { hideBin } from "yargs/helpers"

const cli = yargs(hideBin(process.argv))
  .scriptName("opencode")
  .version("0.0.1")
  .help("h", "show help")
  .command({
    command: "hello [name]",
    describe: "say hello to someone",
    builder: (yargs) => {
      return yargs.positional("name", {
        describe: "name to greet",
        type: "string",
        default: "World",
      })
    },
    handler: async (args) => {
      console.log(`Hello, ${args.name}!`)
    },
  })
  .command({
    command: "serve",
    describe: "start web server",
    handler: async () => {
      console.log("Server starting...")
    },
  })
  .strict()

try {
  await cli.parse(process.argv.slice(2))
} catch (e) {
  console.error("Error:", e)
  process.exit(1)
}